import os
from functools import partial

import torch
import torch.nn.functional as F
import hdf5plugin
import h5py
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .module import JEPA, ARPredictor, Transpressor, SIGReg

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from transformers import AutoImageProcessor, AutoModel

START_VALUE = -2.0
END_VALUE = -3.0

class ChainDataset(Dataset):
    """A chain of num_frames observations connected by num_frames-1 variable-length action hops.

    Each hop is a sequence of macro-steps, not individual raw actions: every macro-step bundles
    `frameskip` consecutive raw actions concatenated along the feature axis (frameskip=1 recovers
    the old one-raw-action-per-step behavior). A hop's length (in macro-steps) is resampled fresh
    on every __getitem__ call, so the same start index yields different hop lengths across epochs.
    """

    def __init__(self, h5_file, num_frames, max_hop_length, frameskip):
        assert num_frames >= 2, "num_frames must be >= 2 (need at least one hop)"
        assert max_hop_length >= 1, "max_hop_length must be >= 1"
        assert frameskip >= 1, "frameskip must be >= 1"

        self.h5_file = h5_file
        self.num_frames = num_frames
        self.max_hop_length = max_hop_length
        self.frameskip = frameskip
        # h5py file handles aren't fork-safe, so don't keep the one used here around - each
        # DataLoader worker process opens its own lazily, on its first __getitem__ call.
        self._h5 = None

        with h5py.File(h5_file, "r") as f:
            ep_offset = f["ep_offset"][:]
            ep_len = f["ep_len"][:]

        # Every hop needs at least one macro-step, i.e. at least `frameskip` raw actions, so a
        # chain of num_frames-1 hops needs at least (num_frames-1)*frameskip raw actions of room.
        min_raw_span = (self.num_frames - 1) * self.frameskip
        starts = []
        ends = []
        for offset, length in zip(ep_offset, ep_len):
            offset = int(offset)
            length = int(length)
            if length >= min_raw_span + 1:
                episode_end = offset + length - 1
                for i1 in range(offset, episode_end - min_raw_span + 1):
                    starts.append(i1)
                    ends.append(episode_end)
        self.starts = starts
        self.ends = ends

    def __len__(self):
        return len(self.starts)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")

    def _sample_hop_lengths(self, i1, episode_end):
        """Reservation-based sampling: reserves >=1 macro-step (frameskip raw actions) for every
        hop still to come, so the sampled range for each hop length is always non-empty - no
        rejection sampling needed. Lengths are counted in macro-steps, each spanning `frameskip`
        raw actions."""
        n_hops = self.num_frames - 1
        lengths = []
        i_t = i1
        for t in range(n_hops):
            hops_after = n_hops - (t + 1)
            raw_budget = episode_end - i_t
            upper = min(self.max_hop_length, (raw_budget - hops_after * self.frameskip) // self.frameskip)
            length = int(torch.randint(1, upper + 1, (1,)).item())
            lengths.append(length)
            i_t += length * self.frameskip
        return lengths

    def __getitem__(self, idx):
        self._ensure_h5()
        i1 = self.starts[idx]
        episode_end = self.ends[idx]
        hop_lengths_raw = self._sample_hop_lengths(i1, episode_end)  # in macro-steps

        frame_indices = [i1]
        i_t = i1
        for length in hop_lengths_raw:
            i_t += length * self.frameskip
            frame_indices.append(i_t)

        pixels = torch.tensor(self._h5["pixels"][frame_indices], dtype=torch.float32)  # type: ignore

        hop_inputs = []
        hop_targets = []
        hop_lengths = []
        i_t = i1
        for length in hop_lengths_raw:
            raw_actions = torch.tensor(
                self._h5["action"][i_t:i_t + length * self.frameskip], dtype=torch.float32  # type: ignore
            )  # (length*frameskip, action_dim)
            # Chunk every frameskip consecutive raw actions into one macro-step, concatenated
            # along the feature axis: (length*frameskip, action_dim) -> (length, frameskip*action_dim).
            actions = raw_actions.reshape(length, -1)

            start_padding = torch.full((1, actions.shape[-1]), START_VALUE)
            end_padding = torch.full((1, actions.shape[-1]), END_VALUE)
            seq = torch.cat([start_padding, actions, end_padding], dim=0)  # (length+2, frameskip*action_dim)

            hop_inputs.append(seq[:-1])   # START, macro-step_0..macro-step_{length-1}
            hop_targets.append(seq[1:])   # macro-step_0..macro-step_{length-1}, END
            hop_lengths.append(length + 1)  # START + length macro-steps
            i_t += length * self.frameskip

        return {
            "pixels": pixels,
            "hop_inputs": hop_inputs,
            "hop_targets": hop_targets,
            "hop_lengths": torch.tensor(hop_lengths, dtype=torch.long),
        }

def chain_collate_fn(batch, preprocessor):
    pixels = torch.stack([item["pixels"] for item in batch], dim=0)  # (B, T, H, W, C)

    # Runs the real HF preprocessor (resize/crop/rescale/normalize) here, inside the DataLoader
    # worker process, instead of in JEPA.encode_pixels inside the model's forward pass - this
    # CPU-bound step now overlaps with GPU compute rather than blocking it every step.
    B, T, H, W, C = pixels.shape
    images = pixels.reshape(B * T, H, W, C).to(torch.uint8).numpy()
    processed = preprocessor(list(images), return_tensors="pt")["pixel_values"]  # (B*T, C, crop, crop)
    pixels = processed.reshape(B, T, *processed.shape[1:])  # (B, T, C, crop, crop)

    all_inputs = [hop for item in batch for hop in item["hop_inputs"]]
    all_targets = [hop for item in batch for hop in item["hop_targets"]]
    hop_input = pad_sequence(all_inputs, batch_first=True)    # (B*(T-1), max_len, action_dim)
    hop_target = pad_sequence(all_targets, batch_first=True)  # (B*(T-1), max_len, action_dim)
    hop_lengths = torch.cat([item["hop_lengths"] for item in batch], dim=0)  # (B*(T-1),)

    return {
        "pixels": pixels,
        "hop_input": hop_input,
        "hop_target": hop_target,
        "hop_lengths": hop_lengths,
    }

def chain_dataloader(h5_file, num_frames, max_hop_length, frameskip, batch_size, preprocessor, num_workers=0, pin_memory=False, distributed=False):
    dataset = ChainDataset(h5_file, num_frames, max_hop_length, frameskip)
    collate_fn = partial(chain_collate_fn, preprocessor=preprocessor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train, test, train_sampler

# -- Setup --
conf = OmegaConf.load("config/transpressor.yaml")
torch.autograd.set_detect_anomaly(True)

n_epochs = conf.n_epochs
lr = conf.lr
batch_size = conf.batch_size
num_frames = conf.num_frames
max_hop_length = conf.max_hop_length
frameskip = conf.frameskip
num_workers = conf.num_workers
log_to_wandb = conf.log_to_wandb
datapath = conf.datapath

device = conf.device
sigreg_term: SIGReg | None = None


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", str(conf.world_size)))
    rank = int(os.environ.get("RANK", "0"))
    # torchrun sets LOCAL_RANK to the GPU index within this node; falls back to rank for
    # single-node CPU runs launched without torchrun.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    distributed = world_size > 1

    if distributed and device not in ("cpu", "cuda"):
        print(f"WARNING: Distributed training is only implemented for cpu/cuda backends, but device is set to {device!r}.")

    if distributed:
        backend = "nccl" if device == "cuda" else "gloo"
        if device == "cuda":
            torch.cuda.set_device(local_rank)
        master_port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{master_port}",
            rank=rank,
            world_size=world_size,
        )

    training_device = torch.device(f"cuda:{local_rank}") if device == "cuda" else torch.device(device)
    return distributed, rank, training_device


def reduce_loss(total_loss, batch_count, training_device):
    values = torch.tensor([total_loss, batch_count], dtype=torch.float64, device=training_device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (values[0] / values[1]).item()

def chain_forward(model, batch, weights, stage="train"):
    """Run a chain batch through JEPA and compute the combined loss:
    per-hop action reconstruction, chain observation prediction, and two SIGReg terms.
    """

    if sigreg_term is None:
        raise RuntimeError("SIGReg must be initialized before training")

    if stage == "train":
        model.train()
    elif stage == "val":
        model.eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    out = model(batch["pixels"], batch["hop_input"], batch["hop_lengths"])

    padded_width = batch["hop_target"].shape[1]
    target_valid = torch.arange(padded_width, device=batch["hop_lengths"].device).unsqueeze(0) \
        < batch["hop_lengths"].unsqueeze(1)
    valid = target_valid.unsqueeze(-1).expand_as(out["decoded_actions"])
    hop_mse_loss = (out["decoded_actions"] - batch["hop_target"]).square().masked_select(valid).mean()

    chain_mse_loss = F.mse_loss(out["pred"], out["target"])
    # (T-1, B, D_cond) / (T, B, D): groups by chain position, giving genuine B-sample groups.
    action_sigreg_loss = sigreg_term(out["hop_emb"].transpose(0, 1))
    obs_sigreg_loss = sigreg_term(out["obs"].transpose(0, 1))

    loss = (
        weights.hop_mse * hop_mse_loss
        + weights.chain_mse * chain_mse_loss
        + weights.action_sigreg * action_sigreg_loss
        + weights.obs_sigreg * obs_sigreg_loss
    )

    return {
        "loss": loss,
        "hop_mse_loss": hop_mse_loss,
        "chain_mse_loss": chain_mse_loss,
        "action_sigreg_loss": action_sigreg_loss,
        "obs_sigreg_loss": obs_sigreg_loss,
    }

def nan_hook(module, inp, out):
    if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
        raise RuntimeError(f"NaN in {module}")
    
if log_to_wandb and int(os.environ.get("RANK", "0")) == 0:
    # Initialize wandb
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="anuragkompalli",
        # Set the wandb project where this run will be logged.
        project="SequenceCompression",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "Transformer",
            "dataset": conf.datapath,
            "epochs": n_epochs,
            "num_frames": num_frames,
            "max_hop_length": max_hop_length,
            "batch_size": batch_size,
            "loss_weights": dict(conf.loss_weights),
            "transpressor_input_dim": conf.transpressor.input_dim,
            "transpressor_hidden_dim": conf.transpressor.hidden_dim,
            "transpressor_condition_dim": conf.transpressor.condition_dim,
            "transpressor_depth": conf.transpressor.depth,
            "transpressor_heads": conf.transpressor.heads,
            "transpressor_dim_head": conf.transpressor.dim_head,
            "transpressor_mlp_dim": conf.transpressor.mlp_dim,
            "transpressor_dropout": conf.transpressor.dropout,
            "transpressor_output_proj": conf.transpressor.output_proj,
            "ar_predictor_hidden_dim": conf.ar_predictor.hidden_dim,
            "ar_predictor_condition_dim": conf.ar_predictor.condition_dim,
            "ar_predictor_depth": conf.ar_predictor.depth,
            "ar_predictor_heads": conf.ar_predictor.heads,
            "ar_predictor_dim_head": conf.ar_predictor.dim_head,
            "ar_predictor_mlp_dim": conf.ar_predictor.mlp_dim,
            "ar_predictor_dropout": conf.ar_predictor.dropout,
            "pixel_preprocessor_model_name": conf.pixel_preprocessor.model_name,
            "pixel_encoder_model_name": conf.pixel_encoder.model_name,

        },
    )

# -- Training loop -- 
def train():
    global sigreg_term
    distributed, rank, training_device = setup_distributed()
    is_main_process = rank == 0
    if is_main_process:
        print(f"Training on {training_device}")

    pixel_preprocessor = AutoImageProcessor.from_pretrained(conf.pixel_preprocessor.model_name)
    pixel_encoder = AutoModel.from_pretrained(conf.pixel_encoder.model_name)

    train_loader, val_loader, train_sampler = chain_dataloader(
        conf.datapath,
        num_frames=num_frames,
        max_hop_length=max_hop_length,
        frameskip=frameskip,
        batch_size=batch_size,
        preprocessor=pixel_preprocessor,
        num_workers=num_workers,
        pin_memory=(training_device.type == "cuda"),
        distributed=distributed,
    )

    # -- Model definition --
    action_encoder = Transpressor(
        # conf.transpressor.input_dim is the raw per-timestep action dim; each macro-step
        # bundles frameskip of them concatenated, so the encoder's real input is frameskip times wider.
        input_dim=frameskip * conf.transpressor.input_dim,
        hidden_dim=conf.transpressor.hidden_dim,
        condition_dim=conf.transpressor.condition_dim,
        depth=conf.transpressor.depth,
        heads=conf.transpressor.heads,
        dim_head=conf.transpressor.dim_head,
        mlp_dim=conf.transpressor.mlp_dim,
        sequence_dim=max_hop_length + 1,
        output_proj=conf.transpressor.output_proj
    ).to(training_device)

    predictor = ARPredictor(
        # x for the chain predictor is the pixel encoder's own CLS-pooled embedding, so its
        # feature dim must track the pixel encoder's hidden size, not conf.ar_predictor.input_dim
        # (which is the action dim and unrelated to this path).
        input_dim=pixel_encoder.config.hidden_size,
        hidden_dim=conf.ar_predictor.hidden_dim,
        condition_dim=conf.ar_predictor.condition_dim,
        depth=conf.ar_predictor.depth,
        heads=conf.ar_predictor.heads,
        dim_head=conf.ar_predictor.dim_head,
        mlp_dim=conf.ar_predictor.mlp_dim,
        dropout=conf.ar_predictor.dropout,
        # The chain predictor attends over num_frames-1 hop-conditioned observation positions.
        sequence_dim=num_frames - 1,
        # ARPredictor is conditioned on encoded_actions, i.e. the action encoder's own
        # output, so its conditioning-input dimensionality must track the same output_proj
        # flag that determines that output's shape (condition_dim vs hidden_dim).
        out_proj=conf.transpressor.output_proj,
    ).to(training_device)

    model = JEPA(
        pixel_encoder=pixel_encoder,
        action_encoder=action_encoder,
        predictor=predictor
    ).to(device=training_device)

    if distributed:
        if training_device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[training_device.index], output_device=training_device.index)
        else:
            model = DistributedDataParallel(model)

    sigreg_term = SIGReg().to(device=training_device)

    for m in model.modules():
        m.register_forward_hook(nan_hook)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    global_step = 0
    for epoch in range(n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # --- Training ---
        train_loss = 0.0
        train_iterator = tqdm(train_loader, desc="Training", disable=not is_main_process)
        for batch in train_iterator:
            batch = {k: v.to(training_device) for k, v in batch.items()}

            optimizer.zero_grad()
            preds = chain_forward(model, batch, conf.loss_weights, stage="train")
            loss = preds["loss"]

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if is_main_process and log_to_wandb:
                run.log({
                    "train/loss": loss.item(),
                    "train/hop_mse_loss": preds["hop_mse_loss"].item(),
                    "train/chain_mse_loss": preds["chain_mse_loss"].item(),
                    "train/action_sigreg_loss": preds["action_sigreg_loss"].item(),
                    "train/obs_sigreg_loss": preds["obs_sigreg_loss"].item(),
                }, step=global_step)
            global_step += 1

        # --- Validation ---
        val_loss = 0.0
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validation", disable=not is_main_process)
            for batch in val_iterator:
                batch = {k: v.to(training_device) for k, v in batch.items()}

                preds = chain_forward(model, batch, conf.loss_weights, stage="val")
                val_loss += preds["loss"].item()

        train_loss = reduce_loss(train_loss, len(train_loader), training_device)
        val_loss = reduce_loss(val_loss, len(val_loader), training_device)

        if is_main_process and log_to_wandb:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr
            })
        elif is_main_process:
            print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Save the model checkpoint
        if is_main_process:
            state_dict = model.state_dict()
            if distributed:
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in state_dict.items()
                }

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(state_dict, f"checkpoints/lewm_epoch_{epoch}.pt")

    if distributed:
        dist.destroy_process_group()
        
if __name__ == "__main__":
    if device == "mps":
        assert torch.mps.is_available(), "MPS is not available!"
    elif device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available!"
    train()
