import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision.transforms.v2 import ToImage, ToDtype, Normalize, Resize, Compose
import torch.nn.functional as F
import hdf5plugin
import h5py
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .module import JEPA, ARPredictor, NaiveActionEmbedder, SIGReg, vit_hf

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from transformers import AutoImageProcessor, AutoModel, AutoConfig
from huggingface_hub import HfApi

# Actions are normalized to mean 0 / unit variance in ChainDataset (raw action std is ~0.2,
# so the normalized range is ~5x wider than raw), so these sentinels sit further out than the
# pre-normalization -2.0/-3.0 to stay outside the normalized action range.
START_VALUE = -2
END_VALUE = -3

class ActionDataset(Dataset):
    """A chain of num_frames observations connected by num_frames-1 variable-length action hops.

    Each hop is a sequence of macro-steps, not individual raw actions: every macro-step bundles
    `frameskip` consecutive raw actions concatenated along the feature axis (frameskip=1 recovers
    the old one-raw-action-per-step behavior). A hop's length (in macro-steps) is resampled fresh
    on every __getitem__ call, so the same start index yields different hop lengths across epochs.
    """

    def __init__(self, h5_file, num_frames, frameskip, preprocessor):
        assert num_frames >= 2, "num_frames must be >= 2 (need at least one hop)"
        assert frameskip >= 1, "frameskip must be >= 1"

        self.h5_file = h5_file
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.preprocessor = preprocessor
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
        for offset, length in zip(ep_offset, ep_len): # type: ignore
            offset = int(offset)
            length = int(length)

            window = num_frames * self.frameskip
            # Only keep windows that fit entirely inside the episode - most episode lengths
            # aren't multiples of `window`, and a trailing partial window would read past the
            # episode's end (silently pulling frames from the next episode, or crashing at the
            # very end of the file).
            for i in range(0, length - window + 1, window):
                starts.append(offset + i)
                ends.append(offset + i + window)

        self.starts = starts
        self.ends = ends

    def __len__(self):
        return len(self.starts)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")

    def __getitem__(self, idx):
        self._ensure_h5()
        i1 = self.starts[idx]
        episode_end = self.ends[idx]

        raw_actions = torch.tensor(
            self._h5["action"][i1:episode_end], dtype=torch.float32 # type: ignore
        )

        actions = raw_actions.reshape(-1, raw_actions.shape[-1]*self.frameskip)

        # Read the whole contiguous span once and subsample in numpy, rather than indexing
        # h5py with a list - on this blosc-compressed, chunked dataset, list/fancy indexing
        # forces slow point-selection reads (~40x slower per item than a contiguous read).
        pixels_block = torch.tensor(self._h5["pixels"][i1:episode_end][::self.frameskip]) # type: ignore

        # ToImage() only permutes HWC -> CHW for raw numpy/PIL input; since pixels_block is
        # already a torch.Tensor here, it would otherwise pass through unpermuted and break
        # Normalize/Resize downstream (which expect channels-first).
        pixels_block = pixels_block.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        pixels = self.preprocessor(pixels_block) # (T, C, H, W)

        return {
            "pixels": pixels,
            "actions": actions
        }

def action_dataloader(h5_file, num_frames, frameskip, batch_size, preprocessor, num_workers=0, pin_memory=False, distributed=False):
    dataset = ActionDataset(h5_file, num_frames, frameskip, preprocessor=preprocessor)

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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train, test, train_sampler

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
            # raw_actions = torch.tensor(f["action"][:], dtype=torch.float32)

        # Per-dimension mean/std over every raw action in the file, used to normalize actions
        # to mean 0 / unit variance in __getitem__. Computed once here (not per-hop) so every
        # split drawn from this dataset (see random_split in chain_dataloader) shares the same
        # statistics rather than each seeing its own.
        # self.action_mean = raw_actions.mean(dim=0)
        # self.action_std = raw_actions.std(dim=0).clamp_min(1e-6)

        # Every hop needs at least one macro-step, i.e. at least `frameskip` raw actions, so a
        # chain of num_frames-1 hops needs at least (num_frames-1)*frameskip raw actions of room.
        min_raw_span = (self.num_frames - 1) * self.frameskip
        starts = []
        ends = []
        for offset, length in zip(ep_offset, ep_len): # type: ignore
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

        # Read the whole contiguous span once and index into the resulting numpy array,
        # rather than indexing h5py with a list - on this blosc-compressed, chunked dataset,
        # list/fancy indexing forces slow point-selection reads (~40x slower per item than a
        # contiguous read), even though the frames themselves land in the same chunk(s).
        pixels_block = self._h5["pixels"][i1:frame_indices[-1] + 1] # type: ignore
        relative_indices = [i - i1 for i in frame_indices]
        pixels = torch.tensor(pixels_block[relative_indices], dtype=torch.float32)

        hop_inputs = []
        hop_targets = []
        hop_lengths = []
        i_t = i1
        for length in hop_lengths_raw:
            raw_actions = torch.tensor(
                self._h5["action"][i_t:i_t + length * self.frameskip], dtype=torch.float32 # type: ignore
            )  # (length*frameskip, action_dim)
            # raw_actions = (raw_actions - self.action_mean) / self.action_std
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
wd = os.path.dirname(os.path.abspath(__file__))
conf = OmegaConf.load("config/transpressor_baseline.yaml")

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

push_to_hub = conf.get("push_to_hub", False)
hf_repo_id = conf.get("hf_repo_id", None)


def upload_checkpoint(api, repo_id, local_path):
    filename = os.path.basename(local_path)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"checkpoints/{filename}",
            repo_id=repo_id,
            repo_type="model",
        )
    except Exception as e:
        print(f"WARNING: failed to upload {filename} to {repo_id}: {e}")


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
    values = torch.tensor([total_loss, batch_count], dtype=torch.float32, device=training_device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (values[0] / values[1]).item()

def model_forward(model, batch, weights, stage="train"):
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

    out = model(batch["pixels"], batch["actions"])

    prediction_mse = weights.prediction_weight * F.mse_loss(out["pred"], out["target"], reduction="mean")

    prediction_sigreg = weights.prediction_sigreg_weight * sigreg_term(out["obs"].transpose(0, 1))

    loss = (
        prediction_mse + prediction_sigreg
    )

    return {
        "loss": loss,
        "prediction_loss": prediction_mse,
        "prediction_sigreg": prediction_sigreg
    }

def nan_hook(module, inp, out):
    if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
        raise RuntimeError(f"NaN in {module}")

import datetime

# -- Training loop --
def train():
    global sigreg_term
    distributed, rank, training_device = setup_distributed()
    is_main_process = rank == 0
    if is_main_process:
        print(f"Training on {training_device}")

    hf_api = None
    hf_executor = None
    if push_to_hub and is_main_process:
        if not hf_repo_id:
            print("WARNING: push_to_hub is enabled but hf_repo_id is not set; skipping hub sync")
        else:
            hf_api = HfApi()
            hf_api.create_repo(repo_id=hf_repo_id, repo_type="model", private=True, exist_ok=True)
            # Single worker keeps uploads in epoch order without blocking the training loop.
            hf_executor = ThreadPoolExecutor(max_workers=1)

    run = None
    if log_to_wandb and is_main_process:
        # Initialize wandb. This must stay inside train() (called only from the
        # `if __name__ == "__main__"` guard below), not at module level - on macOS, DataLoader
        # worker subprocesses (num_workers > 0) re-execute this module's top-level code when
        # they spawn, so a module-level wandb.init() call fires again in every worker process,
        # producing a new run per worker on every script launch.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="anuragkompalli",
            # Set the wandb project where this run will be logged.
            project="SequenceCompression",
            name=f"SequenceCompression_Baseline_{datetime.date.today()}",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                "architecture": "Transformer",
                "dataset": conf.datapath,
                "epochs": n_epochs,
                "num_frames": num_frames,
                "max_hop_length": max_hop_length,
                "batch_size": batch_size,
                **(conf.__dict__)
            },
        )

    # Taken from original le-wm codebase
    pixel_preprocessor =  Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 
            Resize(size=224),
        ]
    )
    
    pixel_encoder = vit_hf("tiny", patch_size=14)

    train_loader, val_loader, train_sampler = action_dataloader(
        conf.datapath,
        num_frames=num_frames,
        frameskip=frameskip,
        batch_size=batch_size,
        preprocessor=pixel_preprocessor,
        num_workers=num_workers,
        pin_memory=(training_device.type == "cuda"),
        distributed=distributed,
    )

    # -- Model definition --
    action_encoder = NaiveActionEmbedder(emb_dim=conf.naive_action_embedder.embed_dim).to(training_device)

    predictor = ARPredictor(
        input_dim=conf.ar_predictor.input_dim,
        hidden_dim=conf.ar_predictor.hidden_dim,
        condition_dim=conf.ar_predictor.condition_dim,
        depth=conf.ar_predictor.depth,
        heads=conf.ar_predictor.heads,
        dim_head=conf.ar_predictor.dim_head,
        mlp_dim=conf.ar_predictor.mlp_dim,
        dropout=conf.ar_predictor.dropout,
        sequence_dim=num_frames - 1,
        out_proj=conf.transpressor.output_proj,
    ).to(training_device)

    model = JEPA(
        pixel_encoder=pixel_encoder,
        action_encoder=action_encoder,
        predictor=predictor
    ).to(device=training_device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameter count: {trainable_params}")

    if distributed:
        if training_device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[training_device.index], output_device=training_device.index)
        else:
            model = DistributedDataParallel(model)

        print("DDP set up")

    sigreg_term = SIGReg().to(device=training_device)

    # for m in model.modules():
    #     m.register_forward_hook(nan_hook)

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
            preds = model_forward(model, batch, conf.loss_weights, stage="train")
            loss = preds["loss"]

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if is_main_process and log_to_wandb:
                run.log({
                    "train/loss": loss.item(),
                    "train/prediction_loss": preds["prediction_loss"].item(),
                    "train/prediction_sigreg": preds["prediction_sigreg"].item(),
                }, step=global_step)
            global_step += 1

        # --- Validation ---
        val_loss = 0.0
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validation", disable=not is_main_process)
            for batch in val_iterator:
                batch = {k: v.to(training_device) for k, v in batch.items()}

                preds = model_forward(model, batch, conf.loss_weights, stage="val")
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
            ckpt_path = f"checkpoints/lewm_epoch_{epoch}.pt"
            torch.save(state_dict, ckpt_path)

            if hf_executor is not None:
                hf_executor.submit(upload_checkpoint, hf_api, hf_repo_id, ckpt_path)

    if hf_executor is not None:
        # Block until the last submitted upload finishes so the process doesn't exit mid-upload.
        hf_executor.shutdown(wait=True)

    if distributed:
        dist.destroy_process_group()
        
if __name__ == "__main__":
    if device == "mps":
        assert torch.mps.is_available(), "MPS is not available!"
    elif device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available!"
    train()
