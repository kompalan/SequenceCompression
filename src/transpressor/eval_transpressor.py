import argparse
import re
from pathlib import Path

import h5py
import torch
from omegaconf import OmegaConf

from .module import Transpressor


CHECKPOINT_PATTERN = re.compile(r"transpressor_epoch_(\d+)\.pt$")


def latest_checkpoint(checkpoint_dir):
    checkpoints = []
    for path in Path(checkpoint_dir).glob("transpressor_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        raise FileNotFoundError(f"No transpressor checkpoints found in {checkpoint_dir}")
    return max(checkpoints)[1]


def build_model(config, checkpoint, device):
    model = Transpressor(
        input_dim=config.transpressor_input_dim,
        hidden_dim=config.transpressor_hidden_dim,
        output_dim=config.transpressor_output_dim,
        depth=config.transpressor_depth,
        heads=config.transpressor_heads,
        dim_head=config.transpressor_dim_head,
        mlp_dim=config.transpressor_mlp_dim,
        dropout=config.transpressor_dropout,
        sequence_dim=config.context_length,
        output_proj=config.transpressor_output_proj,
    )
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.inference_mode()
def evaluate(model, h5_path, context_length, batch_size, device):
    total_squared_error = 0.0
    total_values = 0
    step_squared_error = torch.zeros(context_length, dtype=torch.float64)
    step_values = torch.zeros(context_length, dtype=torch.float64)
    chunks = []

    def evaluate_batch(batch):
        nonlocal total_squared_error, total_values
        if not batch:
            return

        actions = torch.stack([item[0] for item in batch]).to(device)
        lengths = torch.tensor([item[1] for item in batch], device=device)
        start = torch.full(
            (len(batch), 1, actions.shape[-1]), -2.0, device=device
        )
        encoded_input = torch.cat([start, actions], dim=1)
        encoded = model.encode(encoded_input)

        sequence = start
        predictions = []
        for _ in range(context_length):
            prediction = model.decode(sequence, encoded)[:, -1:, :]
            predictions.append(prediction)
            sequence = torch.cat([sequence, prediction], dim=1)
        predictions = torch.cat(predictions, dim=1)

        errors = (predictions - actions).square()
        valid = torch.arange(context_length, device=device)[None, :] < lengths[:, None]
        valid = valid.unsqueeze(-1)
        total_squared_error += errors.masked_select(valid).sum().item()
        total_values += valid.sum().item() * actions.shape[-1]

        for step in range(context_length):
            step_valid = valid[:, step].expand_as(errors[:, step])
            step_squared_error[step] += errors[:, step].masked_select(step_valid).sum().item()
            step_values[step] += step_valid[:, 0].sum().item() * actions.shape[-1]

    with h5py.File(h5_path, "r") as data:
        actions = data["action"]
        for offset, episode_length in zip(data["ep_offset"][:], data["ep_len"][:]):
            offset = int(offset)
            episode_length = int(episode_length)
            episode = torch.from_numpy(actions[offset : offset + episode_length])

            for start in range(0, episode_length, context_length):
                length = min(context_length, episode_length - start)
                chunk = torch.full(
                    (context_length, episode.shape[-1]), -3.0, dtype=torch.float32
                )
                chunk[:length] = episode[start : start + length]
                chunks.append((chunk, length))
                if len(chunks) == batch_size:
                    evaluate_batch(chunks)
                    chunks.clear()

    evaluate_batch(chunks)
    return (
        total_squared_error / total_values,
        step_squared_error / step_values.clamp_min(1),
        total_values,
    )


def run(data_path="data/pusht_expert_train.h5", checkpoint_dir="checkpoints", device=None):
    config = OmegaConf.load("config/transpressor.yaml")
    device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    checkpoint = latest_checkpoint(checkpoint_dir)
    model = build_model(config, checkpoint, device)
    mse, step_mse, value_count = evaluate(
        model,
        data_path,
        context_length=config.context_length,
        batch_size=config.batch_size,
        device=device,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Evaluated action values: {value_count}")
    print(f"Rollout MSE: {mse:.8f}")
    for step, value in enumerate(step_mse.tolist(), start=1):
        print(f"Step {step} MSE: {value:.8f}")
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transpressor rollouts.")
    parser.add_argument("--data", default="data/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", choices=("cpu", "mps"), default=None)
    args = parser.parse_args()
    run(args.data, args.checkpoint_dir, args.device)
