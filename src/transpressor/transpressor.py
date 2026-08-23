import torch
import torch.nn.functional as F
import h5py

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .module import Transpressor

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

def action_dataloader(h5_file, context_length, batch_size):
    dataset = ActionDataset(h5_file, context_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train, test

def compressor_forward(model, input, target, stage="train"):
    """Encode a sequence into a summary vector and train a decoder to reconstruct it."""

    if stage == "train":
        model.train()
    elif stage == "val":
        model.eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    encoded = model.encode(input)
    decoded = model.decode(input, encoded)

    # Predict the next action token from the summary plus the preceding context.
    # Take the first seq_len-1 tokens and compare it to a shifted version of actions[1:]
    loss = F.mse_loss(decoded, target)

    output = {
        "actions": input,
        "compressed": encoded,
        "decompressed": decoded,
        "loss": loss,
    }

    return output

conf = OmegaConf.load("config/transpressor.yaml")

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

if log_to_wandb:
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

def train():
    train_loader, val_loader = action_dataloader("data/pusht_expert_train.h5", context_length=context_length, batch_size=batch_size)
    
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
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)


    for epoch in range(n_epochs):
        # --- Training ---
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input = batch["input"].flatten(0, 1).to(device)
            target = batch["target"].flatten(0, 1).to(device)

            optimizer.zero_grad()
            preds = compressor_forward(model, input, target, stage="train")
            loss = preds["loss"]

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input = batch["input"].flatten(0, 1).to(device)
                target = batch["target"].flatten(0, 1).to(device)
                preds = compressor_forward(model, input, target, stage="val")
                val_loss += preds["loss"].item()

        if log_to_wandb:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "learning_rate": lr
            })
        else:
            print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")
        
if __name__ == "__main__":
    train()
