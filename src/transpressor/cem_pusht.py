"""CEM planning over JEPA's latent action space, rolled out on the real PushT env as MPC.

CEM (via evotorch) searches directly in the D_cond-dimensional latent action space that
JEPA.rollout consumes (see module.py's docstring: raw N(0, I) draws are valid inputs
there because of the SIGReg training term) rather than in raw 2D PushT actions. Every
`actions_per_plan` real control steps: CEM plans a `plan_steps`-hop latent trajectory from
the CURRENT real observation, `actions_per_plan` raw actions are decoded from the winning
plan's first hop and executed on the physical env, and the rest of the plan is discarded -
receding-horizon MPC with a configurable execution horizon (actions_per_plan=1 is the
classic replan-every-step case). Frames from every step are recorded to an mp4.
"""

import argparse
import logging
import re
from pathlib import Path

import evotorch
import gymnasium as gym
import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0
import imageio
import numpy as np
import torch
from evotorch.algorithms import CEM
from omegaconf import OmegaConf
from transformers import AutoImageProcessor, AutoModel

from .module import JEPA, ARPredictor, Transpressor

# MPC replans every step, i.e. one evotorch Problem per step - silence its per-instance
# INFO banner so a multi-step rollout doesn't spam four log lines per control step.
logging.getLogger("evotorch").setLevel(logging.WARNING)

START_VALUE = -2.0

# PushT's agent is a PD-controlled kinematic point: env.step(target) drives it toward an
# absolute pixel-space target in [0, 512]^2. The h5 dataset (and hence everything JEPA's
# action encoder was trained on) instead stores actions as that target expressed relative
# to the agent's current position and divided by 100 - confirmed by simulating the env's
# own PD loop (k_p=100, k_v=20) against the dataset's (state, action, next_state) triples,
# which reproduces next_state to within 1e-5 using exactly this constant.
ACTION_TO_TARGET_SCALE = 100.0


def latest_checkpoint(checkpoint_dir):
    pattern = re.compile(r"lewm_epoch_(\d+)\.pt$")
    checkpoints = []
    for path in Path(checkpoint_dir).glob("lewm_epoch_*.pt"):
        match = pattern.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        raise FileNotFoundError(f"No lewm checkpoints found under {checkpoint_dir}")
    return max(checkpoints)[1]


def build_jepa(config, checkpoint, device):
    pixel_encoder = AutoModel.from_pretrained(config.pixel_encoder.model_name)

    action_encoder = Transpressor(
        input_dim=config.frameskip * config.transpressor.input_dim,
        hidden_dim=config.transpressor.hidden_dim,
        condition_dim=config.transpressor.condition_dim,
        depth=config.transpressor.depth,
        heads=config.transpressor.heads,
        dim_head=config.transpressor.dim_head,
        mlp_dim=config.transpressor.mlp_dim,
        sequence_dim=config.max_hop_length + 1,
        output_proj=config.transpressor.output_proj,
    )

    predictor = ARPredictor(
        input_dim=pixel_encoder.config.hidden_size,
        hidden_dim=config.ar_predictor.hidden_dim,
        condition_dim=config.ar_predictor.condition_dim,
        depth=config.ar_predictor.depth,
        heads=config.ar_predictor.heads,
        dim_head=config.ar_predictor.dim_head,
        mlp_dim=config.ar_predictor.mlp_dim,
        dropout=config.ar_predictor.dropout,
        sequence_dim=config.num_frames - 1,
        out_proj=config.transpressor.output_proj,
    )

    model = JEPA(pixel_encoder=pixel_encoder, action_encoder=action_encoder, predictor=predictor)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def preprocess_frame(preprocessor, frame, device):
    """frame: (H, W, 3) uint8 -> (1, C, crop, crop) preprocessed pixel tensor."""
    return preprocessor([frame], return_tensors="pt")["pixel_values"].to(device)


def render_goal_frame(env, goal_pose):
    """Snapshot the env with the block placed exactly on its goal pose."""
    obs, _ = env.reset(options={"reset_to_state": [256.0, 450.0, *goal_pose]})
    return obs["pixels"]


@torch.inference_mode()
def plan_with_cem(jepa, pixels_hist, goal_pixels, plan_steps, cond_dim, args, device):
    def fitness(solutions):
        popsize = solutions.shape[0]
        latent_actions = solutions.view(1, popsize, plan_steps, cond_dim)
        cost = jepa.get_cost(pixels_hist, goal_pixels, latent_actions)
        # An undertrained predictor can occasionally diverge for a given latent draw; treat
        # that candidate as maximally bad rather than letting a NaN poison CEM's mean/stdev.
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6)
        return cost.view(popsize)

    problem = evotorch.Problem(
        "min",
        fitness,
        solution_length=plan_steps * cond_dim,
        initial_bounds=(-1.0, 1.0),
        dtype=torch.float32,
        device=device,
        vectorized=True,
    )
    searcher = CEM(
        problem,
        popsize=args.popsize,
        parenthood_ratio=args.parenthood_ratio,
        stdev_init=args.stdev_init,
    )
    searcher.run(args.generations)

    best = searcher.status["pop_best"]
    return best.values.view(plan_steps, cond_dim), float(best.evals[0])


@torch.inference_mode()
def decode_hop(jepa, latent_condition, num_actions, action_dim, device):
    """Autoregressively decode `num_actions` raw actions conditioned on one latent hop."""
    c = latent_condition.view(1, 1, -1)
    sequence = torch.full((1, 1, action_dim), START_VALUE, device=device)
    for _ in range(num_actions):
        prediction = jepa.decode_actions(sequence, c)[:, -1:, :]
        sequence = torch.cat([sequence, prediction], dim=1)
    return sequence[0, 1:]  # (num_actions, action_dim)


def run(args):
    config = OmegaConf.load(args.config)
    device = torch.device(args.device)

    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint}")
    jepa = build_jepa(config, checkpoint, device)
    preprocessor = AutoImageProcessor.from_pretrained(config.pixel_preprocessor.model_name)

    plan_steps = args.plan_steps or (config.num_frames - 1)
    cond_dim = config.transpressor.condition_dim
    action_dim = config.frameskip * config.transpressor.input_dim

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    goal_pose = (256.0, 256.0, np.pi / 4)
    goal_frame = render_goal_frame(env, goal_pose)
    goal_pixels = preprocess_frame(preprocessor, goal_frame, device)

    obs, _ = env.reset(seed=args.seed)
    agent_pos = obs["agent_pos"]
    frames = [env.render()]

    is_success = False
    step = 0
    while step < args.max_steps:
        # Replan from scratch against the real observation just seen - T_hist=1, no reuse
        # of the previous step's imagined trajectory. This is what makes it MPC rather than
        # the earlier plan-once-and-execute version: only this replan's executed actions
        # survive, everything after them in the plan is discarded and replanned fresh.
        pixels_hist = preprocess_frame(preprocessor, obs["pixels"], device).unsqueeze(1)  # (1, 1, C, H, W)
        plan, cost = plan_with_cem(jepa, pixels_hist, goal_pixels, plan_steps, cond_dim, args, device)

        # actions_per_plan controls the execution horizon (how many actions get committed
        # before the next replan) independently of plan_steps (how far CEM looks ahead when
        # scoring a plan) - 1 is pure per-step MPC, actions_per_plan >= plan_steps recovers
        # the earlier plan-once-and-execute-in-full behavior.
        chunk_len = min(args.actions_per_plan, args.max_steps - step)
        raw_actions = decode_hop(jepa, plan[0], chunk_len, action_dim, device)

        for raw_action in raw_actions:
            # An undertrained (or out-of-distribution-latent) decoder can occasionally emit a
            # blown-up or NaN action; pymunk propagates a NaN position permanently once it
            # appears, so sanitize before it ever reaches env.step.
            raw_action = np.nan_to_num(raw_action.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            target = np.clip(agent_pos + ACTION_TO_TARGET_SCALE * raw_action, 0.0, 512.0)

            obs, _, terminated, _, info = env.step(target)
            agent_pos = obs["agent_pos"]
            frames.append(env.render())
            step += 1
            print(f"step {step:3d} | CEM cost {cost:10.4f} | coverage {info['coverage']:.3f}")

            if terminated:
                is_success = True
                break
        if is_success:
            break

    print(f"Success: {is_success} | final coverage: {info['coverage']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps)
    print(f"Saved rollout ({len(frames)} frames) to {args.out}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="CEM-plan a JEPA action sequence and roll it out on PushT.")
    parser.add_argument("--config", default="config/transpressor.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plan-steps", type=int, default=None, help="CEM planning horizon in latent hops (default: num_frames - 1)")
    parser.add_argument("--max-steps", type=int, default=40, help="Total real env steps to execute across the whole rollout")
    parser.add_argument("--actions-per-plan", type=int, default=1, help="Actions to execute from each CEM plan before replanning (1 = pure per-step MPC)")
    parser.add_argument("--popsize", type=int, default=128)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--parenthood-ratio", type=float, default=0.1)
    parser.add_argument("--stdev-init", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", default="rollout.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
