import os
from functools import partial
from pathlib import Path

import hydra

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .module import SIGReg, modulate


@hydra.main(version_base=None, config_path="./config/train", config_name="transpressor")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    # del dataset_cfg["num_steps"]
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    transpressor = swm.wm.utils.load_pretrained("transpressor/weights_epoch_2.pt")
    transpressor = transpressor.to("cuda")
    transpressor = transpressor.eval()
    transpressor.requires_grad_(False)
    
    mses = []
    for info in val:
        actions = info["action"][:, :, :].to("cuda")
        batch_size, seq_len, act_dim = actions.shape
        start_padding = torch.full((batch_size, 1, act_dim), -2.0, device="cuda")
        end_padding = torch.full((batch_size, 1, act_dim), -3.0, device="cuda")
        actions = torch.cat([start_padding, actions, end_padding], dim=1)
        batch_size, seq_len, act_dim = actions.shape

        encoded = transpressor.encode(actions)
        decoded = transpressor.decode(actions, encoded)
        # print((decoded[:, :-1] - actions[:, 1:]).pow(2).mean()) 
        print(F.mse_loss(decoded[:, :-1], actions[:, 1:]))
        def generate(start, encoded):
            sequence = start
            
            # while not torch.all(torch.where(sequence[:, -1, :] < -3, True, False)):
            for i in range(7):
                sequence = torch.cat([sequence, transpressor.decode(sequence, encoded)[:, -1, :].unsqueeze(1)], dim=1)

            return sequence
        
        final = generate(start_padding, encoded)
        print(torch.stack([final, actions], dim=1))
        mses.append(F.mse_loss(final, actions))
    
    mses = torch.tensor(mses).mean(dim=-1)
    print(mses)
    return


if __name__ == "__main__":
    run()
