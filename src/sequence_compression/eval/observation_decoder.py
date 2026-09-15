"""Decoder probe: reconstructs images from the JEPA pixel encoder's CLS token.

Trains a small transformer decoder on top of the *frozen* ViT from a trained JEPA
checkpoint, to visualize how much visual information the CLS token retains.
"""
import argparse
from functools import partial
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from sequence_compression.eval.eval import build_model_pipeline
from sequence_compression.module import Block


class CLSDecoder(nn.Module):
    """Attends a grid of learned patch queries to the CLS token, then projects each
    query to a pixel patch and reassembles the patches into an RGB image (MAE-style)."""

    def __init__(
        self,
        embed_dim,
        image_size=224,
        patch_size=16,
        hidden_dim=256,
        depth=4,
        heads=8,
        dim_head=32,
        mlp_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size**2

        self.cls_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [Block(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.patch_head = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, cls_token):
        batch = cls_token.shape[0]
        cls = self.cls_proj(cls_token).unsqueeze(1)  # (B, 1, hidden_dim), conditioning token
        queries = self.query_token.expand(batch, self.pos_embed.shape[1], -1) + self.pos_embed
        x = torch.cat([cls, queries], dim=1)  # (B, 1 + num_patches, hidden_dim)

        for block in self.blocks:
            x = block(x, attn_mask=None)  # full (non-causal) self-attention

        x = self.norm(x)
        patches = self.patch_head(x[:, 1:])  # drop the CLS slot, keep the patch queries

        return rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.grid_size, w=self.grid_size, p1=self.patch_size, p2=self.patch_size, c=3,
        )


class FrameDataset(Dataset):
    """Flat, order-independent view over every frame stored in the chain h5 file."""

    def __init__(self, h5_file, indices=None):
        self.h5_file = h5_file
        with h5py.File(h5_file, "r") as f:
            total = f["pixels"].shape[0]
        self.indices = indices if indices is not None else np.arange(total)
        self._h5 = None

    def __len__(self):
        return len(self.indices)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")

    def __getitem__(self, idx):
        self._ensure_h5()
        return self._h5["pixels"][self.indices[idx]]  # (H, W, C) uint8


def collate_fn(batch, preprocessor):
    return preprocessor(list(batch), return_tensors="pt")["pixel_values"]  # (B, C, H, W)


def unnormalize(pixel_values, preprocessor):
    mean = torch.tensor(preprocessor.image_mean, device=pixel_values.device).view(1, -1, 1, 1)
    std = torch.tensor(preprocessor.image_std, device=pixel_values.device).view(1, -1, 1, 1)
    return (pixel_values * std + mean).clamp(0, 1)


def save_reconstructions(originals, reconstructions, preprocessor, out_path):
    originals = unnormalize(originals, preprocessor).cpu()
    reconstructions = unnormalize(reconstructions, preprocessor).cpu()

    rows = []
    for orig, recon in zip(originals, reconstructions):
        pair = torch.cat([orig, recon], dim=2)  # side by side along width
        rows.append(pair)
    grid = torch.cat(rows, dim=1)  # stacked along height

    array = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(out_path)


def train_probe(args):
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)

    model, preprocessor = build_model_pipeline(cfg, args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    decoder = CLSDecoder(
        embed_dim=model.pixel_encoder.config.hidden_size,
        hidden_dim=args.hidden_dim,
        depth=args.decoder_depth,
        heads=args.decoder_heads,
    ).to(device)
    optimizer = AdamW(decoder.parameters(), lr=args.lr)

    with h5py.File(args.data, "r") as f:
        total_frames = f["pixels"].shape[0]
    # Evenly spaced subsample keeps reads mostly sequential (point-selection into a
    # chunked/compressed h5 dataset is far slower than a contiguous scan).
    num_samples = min(args.max_samples, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples).astype(np.int64)

    val_size = max(1, int(0.02 * num_samples))
    train_indices, val_indices = indices[:-val_size], indices[-val_size:]

    collate = partial(collate_fn, preprocessor=preprocessor)
    train_loader = DataLoader(
        FrameDataset(args.data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FrameDataset(args.data, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0.0
        for pixel_values in tqdm(train_loader, desc=f"epoch {epoch} train"):
            pixel_values = pixel_values.to(device)  # (B, C, H, W), preprocessed

            with torch.no_grad():
                # (B, T=1, C, H, W) in, (B, T=1, hidden_dim) out - one frame per sample
                cls_token = model.pixel_encoder(pixel_values).last_hidden_state[:, 0, :].squeeze(1)

            reconstruction = decoder(cls_token)
            loss = F.mse_loss(reconstruction, pixel_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        decoder.eval()
        val_loss = 0.0
        first_batch = None
        with torch.no_grad():
            for pixel_values in val_loader:
                pixel_values = pixel_values.to(device)
                cls_token = model.pixel_encoder(pixel_values).last_hidden_state[:, 0, :].squeeze(1)
                reconstruction = decoder(cls_token)
                val_loss += F.mse_loss(reconstruction, pixel_values).item()
                if first_batch is None:
                    first_batch = (pixel_values[: args.num_vis], reconstruction[: args.num_vis])

        print(
            f"epoch {epoch}: train_loss={train_loss / len(train_loader):.4f} "
            f"val_loss={val_loss / len(val_loader):.4f}"
        )

        save_reconstructions(*first_batch, preprocessor, output_dir / f"epoch_{epoch}.png")
        torch.save(decoder.state_dict(), output_dir / "decoder.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/transpressor_baseline.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/lewm_epoch_9.pt")
    parser.add_argument("--data", default="data/pusht_expert_train.h5")
    parser.add_argument("--output-dir", default="results/observation_decoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=18000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        cfg = OmegaConf.load(args.config)
        args.device = cfg.device
    return args


if __name__ == "__main__":
    train_probe(parse_args())
