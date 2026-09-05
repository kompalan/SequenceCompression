import os

import torch
import torch.nn.functional as F
import h5py
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .module import Transpressor, SIGReg

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

class ActionDataset(Dataset):
    def __init__(self, h5_file, context_length):
        self.h5_file = h5_file
        self.context_length = context_length
        self._h5 = h5py.File(self.h5_file, "r")

        with h5py.File(h5_file, "r") as f:
            self.length = len(self._h5["action"]) - self.context_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = idx
        actions = torch.tensor(self._h5["action"][i:i + self.context_length], dtype=torch.float32)
        actions = actions.unsqueeze(0).expand(self.context_length, self.context_length, -1)  # Expand to (context_length, context_length, action_dim)

        # Create an upper triangular mask with 1's on the off diagonal and 0's on the diagonal and below
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1).bool().unsqueeze(-1)  # Shape: (context_length, context_length, 1)
        actions = actions.masked_fill(mask, -3.0)  # Fill the masked positions with -3.0

        start_padding = torch.full((actions.shape[1], 1, actions.shape[-1]), -2.0)
        end_padding = torch.full((actions.shape[1], 1, actions.shape[-1]), -3.0)
        input_seq_with_delimiters = torch.cat([start_padding, actions, end_padding], dim=1)

        input = input_seq_with_delimiters[:, :-1, :]
        target = input_seq_with_delimiters[:, 1:, :]
        return {"input": input, "target": target}

def action_dataloader(h5_file, context_length, batch_size, distributed=False):
    dataset = ActionDataset(h5_file, context_length)

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
    )
    test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=True,
    )

    return train, test, train_sampler

# -- Setup --
conf = OmegaConf.load("config/transpressor.yaml")
torch.autograd.set_detect_anomaly(True)

n_epochs = conf.n_epochs
lr = conf.lr
batch_size = conf.batch_size
context_length = conf.context_length
log_to_wandb = conf.log_to_wandb

transpressor_input_dim = conf.transpressor_input_dim
transpressor_hidden_dim = conf.transpressor_hidden_dim
transpressor_output_dim = conf.transpressor_output_dim
transpressor_depth = conf.transpressor_depth
transpressor_heads = conf.transpressor_heads
transpressor_dim_head = conf.transpressor_dim_head
transpressor_mlp_dim = conf.transpressor_mlp_dim
transpressor_dropout = conf.transpressor_dropout
transpressor_output_proj = conf.transpressor_output_proj
device = conf.device
sigreg_term: SIGReg | None = None


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed and device != "cpu":
        raise RuntimeError(
            f"Distributed training is only supported on CPU, but device is configured as {device!r}. "
            "Run a single process to train on MPS."
        )
    if distributed:
        master_port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{master_port}",
            rank=rank,
            world_size=world_size,
        )

    training_device = torch.device(device)
    return distributed, rank, training_device


def reduce_loss(total_loss, batch_count, training_device):
    values = torch.tensor([total_loss, batch_count], dtype=torch.float64, device=training_device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (values[0] / values[1]).item()

def compressor_forward(model, input, target, stage="train"):
    """Encode a sequence into a summary vector and train a decoder to reconstruct it."""

    if sigreg_term is None:
        raise RuntimeError("SIGReg must be initialized before training")

    if stage == "train":
        model.train()
    elif stage == "val":
        model.eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    decoded, encoded = model(input)


    mse_loss = F.mse_loss(decoded, target)
    sigreg_loss = sigreg_term(encoded)
    # Predict the next action token from the summary plus the preceding context.
    # Take the first seq_len-1 tokens and compare it to a shifted version of actions[1:]
    loss = mse_loss + sigreg_loss

    output = {
        "actions": input,
        "compressed": encoded,
        "decompressed": decoded,
        "loss": loss,
        "mse_loss": mse_loss,
        "sigreg_loss": sigreg_loss
    }

    return output

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
            "dataset": "kompallia/pusht_expert_train.h5",
            "epochs": n_epochs,
            "context_length": context_length,
            "batch_size": batch_size,
            "transpressor_input_dim": transpressor_input_dim,
            "transpressor_hidden_dim": transpressor_hidden_dim,
            "transpressor_output_dim": transpressor_output_dim,
            "transpressor_depth": transpressor_depth,
            "transpressor_heads": transpressor_heads,
            "transpressor_dim_head": transpressor_dim_head,
            "transpressor_mlp_dim": transpressor_mlp_dim,
            "transpressor_dropout": transpressor_dropout,
            "transpressor_output_proj": transpressor_output_proj
        },
    )

# -- Training loop -- 
def train():
    global sigreg_term
    distributed, rank, training_device = setup_distributed()
    is_main_process = rank == 0
    if is_main_process:
        print(f"Training on {training_device}")
    train_loader, val_loader, train_sampler = action_dataloader(
        "data/pusht_expert_train.h5",
        context_length=context_length,
        batch_size=batch_size,
        distributed=distributed,
    )
    
    model = Transpressor(
        input_dim=transpressor_input_dim, 
        hidden_dim=transpressor_hidden_dim, 
        output_dim=transpressor_output_dim, 
        depth=transpressor_depth, 
        heads=transpressor_heads,
        dim_head=transpressor_dim_head, 
        mlp_dim=transpressor_mlp_dim, 
        sequence_dim=context_length,
        output_proj=transpressor_output_proj
    ).to(training_device)

    if distributed:
        model = DistributedDataParallel(model)

    sigreg_term = SIGReg().to(device=training_device)

    for m in model.modules():
        m.register_forward_hook(nan_hook)

    optimizer = AdamW(model.parameters(), lr=lr)


    for epoch in range(n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # --- Training ---
        train_loss = 0.0
        train_iterator = tqdm(train_loader, desc="Training", disable=not is_main_process)
        for batch in train_iterator:
            input = batch["input"].flatten(0, 1).to(training_device)
            target = batch["target"].flatten(0, 1).to(training_device)

            optimizer.zero_grad()
            preds = compressor_forward(model, input, target, stage="train")
            loss = preds["loss"]

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        val_loss = 0.0
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validation", disable=not is_main_process)
            for batch in val_iterator:
                input = batch["input"].flatten(0, 1).to(training_device)
                target = batch["target"].flatten(0, 1).to(training_device)
                preds = compressor_forward(model, input, target, stage="val")
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
            torch.save(state_dict, f"checkpoints/transpressor_epoch_{epoch}.pt")

    if distributed:
        dist.destroy_process_group()
        
if __name__ == "__main__":
    assert torch.mps.is_available() == True, "MPS is not available!"
    train()
