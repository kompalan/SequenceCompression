import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import numpy as np
import hdf5plugin
import h5py
import torch
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from sequence_compression.module import JEPA, Transpressor, ARPredictor, NaiveActionEmbedder

def build_model_pipeline(config, checkpoint, device):
    pixel_preprocessor = AutoImageProcessor.from_pretrained(config.pixel_preprocessor.model_name)
    vitconfig = AutoConfig.from_pretrained(config.pixel_encoder.model_name)
    pixel_encoder = AutoModel.from_config(vitconfig)

    action_encoder = NaiveActionEmbedder(emb_dim=config.naive_action_embedder.embed_dim).to(device)

    predictor = ARPredictor(
        input_dim=config.ar_predictor.input_dim,
        hidden_dim=config.ar_predictor.hidden_dim,
        condition_dim=config.ar_predictor.condition_dim,
        depth=config.ar_predictor.depth,
        heads=config.ar_predictor.heads,
        dim_head=config.ar_predictor.dim_head,
        mlp_dim=config.ar_predictor.mlp_dim,
        dropout=config.ar_predictor.dropout,
        sequence_dim=config.num_frames - 1,
        out_proj=config.transpressor.output_proj,
    ).to(device)

    model = JEPA(
        pixel_encoder=pixel_encoder,
        action_encoder=action_encoder,
        predictor=predictor
    ).to(device=device)

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), pixel_preprocessor

def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)

def image_transform(image_processor):
    def transform(image):
        # image = image.transpose(0, 2)
        # print(image.shape)
        return image_processor(image, return_tensors="pt")["pixel_values"]

    return transform

def run(cfg):
    num_envs = 5

    # Create the PushT environment
    world = swm.World("swm/PushT-v1", image_shape=(224, 224), num_envs=num_envs)

    # -- run evaluation
    model, image_processor = build_model_pipeline(cfg, "checkpoints/lewm_epoch_12.pt", device=cfg.device)

    config = swm.PlanConfig(horizon=1, receding_horizon=1, history_len=3, action_block=cfg.frameskip)

    solver = swm.solver.CEMSolver(
        model=model,
        device=cfg.device,

    )

    transform = {
        "pixels": image_transform(image_processor),
        "goal": image_transform(image_processor),
    }

    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, transform=transform
    )

    dataset = swm.data.HDF5Dataset(
        path="data/pusht_expert_train.h5",
        frameskip=cfg.frameskip,
        # "action" is stored per-step (dim 2) but frameskip>1 reshapes it into
        # frameskip*2-wide blocks, which then collides with and overwrites the
        # live rollout's per-step "action" info (shape 2) in world.evaluate,
        # breaking env_pool's stacked-info buffer. We don't need replayed
        # dataset actions here since CEMSolver drives the policy itself.
        keys_to_load=["episode_idx", "pixels", "proprio", "state", "step_idx"],
    )

    starts = torch.randint(0, 18685, (num_envs,))

    results_path = Path("results")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=torch.zeros_like(starts).tolist(),
        goal_offset=5,
        eval_budget=100,
        episodes_idx=starts.tolist(),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / "out"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

def eval():
    conf_path = "config/transpressor_baseline.yaml"
    config = OmegaConf.load(conf_path)
    run(config)

if __name__ == "__main__":
    eval()