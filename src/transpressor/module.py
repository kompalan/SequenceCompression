import os
from functools import partial
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from einops import rearrange

from transformers import AutoImageProcessor, AutoModel

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.knots = knots
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)
    
class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        # self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, attn_mask):
        """
        x : (B, T, D)
        attn_mask : bool, broadcastable to (B, heads, T, T), True = attend
        """
        # x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


def build_causal_padding_mask(seq_len, lengths, device):
    """Combined causal + key-padding mask.

    lengths: (B,) true sequence length per sample. Returns bool (B, 1, T, T), True = attend.
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    key_valid = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
    mask = causal[None, :, :] & key_valid[:, None, :]
    return mask.unsqueeze(1)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, context_dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        self.conditional_proj = (
            nn.Linear(context_dim, dim, bias=True)
            if dim != context_dim
            else nn.Identity()
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0) # type: ignore
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0) # type: ignore

    def forward(self, x, c, attn_mask):
        c = self.conditional_proj(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        condition_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.1,
        sequence_dim=7,
        out_proj=False
    ):
        super().__init__()
        # self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
        )
        
        self.pos_enc = (
            nn.Embedding(2 * sequence_dim, hidden_dim)
        )

        for _ in range(depth):
            self.layers.append(
                Block(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )
            

        self.output_proj = (
            nn.Linear(hidden_dim, condition_dim)
            if out_proj
            else nn.Identity()
        )

        if out_proj:
            self.mlp = MLP(condition_dim, mlp_dim)
        else:
            self.mlp = MLP(hidden_dim, mlp_dim)

    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, action_dim)
        lengths: (batch,) true sequence length per sample; None = fully valid (old behavior)
        """
        batch, seq_len, _ = x.shape

        if lengths is None:
            lengths = torch.full((batch,), seq_len, dtype=torch.long, device=x.device)

        attn_mask = build_causal_padding_mask(seq_len, lengths, x.device)

        x = self.input_proj(x)
        x = x + self.pos_enc(torch.arange(seq_len, device=x.device))

        for block in self.layers:
            x = block(x, attn_mask)

        x = self.output_proj(x)

        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, x.shape[-1])
        last = torch.gather(x, 1, idx)

        return self.mlp(last)
    
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        condition_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.1,
        sequence_dim=7,
        out_proj=False
    ):
        super().__init__()

        if out_proj:
            self.input_norm = nn.LayerNorm(condition_dim)
        else:
            self.input_norm = nn.LayerNorm(hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_dropout = nn.Dropout(0.50)

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
        )
        
        self.pos_enc = (
            nn.Embedding(2 * sequence_dim, hidden_dim)
        )
        
        self.output_proj = (
            nn.Linear(hidden_dim, input_dim)
        )

        for _ in range(depth):
            self.layers.append(
                ConditionalBlock(hidden_dim, condition_dim, heads, dim_head, mlp_dim, dropout=dropout)
            )

        if out_proj:
            self.mlp = MLP(condition_dim, mlp_dim)
        else:
            self.mlp = MLP(hidden_dim, mlp_dim)

    def forward(self, x, c=None, lengths=None):
        """
        x: (batch, sequence_dim, action_dim)
        c: (batch, 1, embed_dim)
        lengths: (batch,) true sequence length per sample; None = fully valid (old behavior)
        """
        batch, seq_len, _ = x.shape

        if lengths is None:
            lengths = torch.full((batch,), seq_len, dtype=torch.long, device=x.device)

        attn_mask = build_causal_padding_mask(seq_len, lengths, x.device)

        c = self.mlp(c)
        c = self.input_norm(c)

        x = self.input_proj(x)
        x = self.input_dropout(x)

        x = x + self.pos_enc(torch.arange(seq_len, device=x.device))

        for block in self.layers:
            x = block(x, c, attn_mask)

        x = self.norm(x)

        x = self.output_proj(x)
            
        return x

class ARPredictor(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            condition_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout=0.1,
            sequence_dim=7,
            out_proj=False,
    ):
        super().__init__()
        self.transformer = TransformerDecoder(
            input_dim, hidden_dim, condition_dim, depth, heads, dim_head, mlp_dim,
            dropout=dropout, sequence_dim=sequence_dim, out_proj=out_proj
        )

    def forward(self, x, c=None):
        return self.transformer(x, c)

class Transpressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        condition_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.1,
        sequence_dim=7,
        output_proj=False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim, hidden_dim, condition_dim, depth, heads, 
            dim_head, mlp_dim, dropout=dropout, 
            sequence_dim=sequence_dim, out_proj=output_proj
        )

        self.decoder = TransformerDecoder(
            input_dim, hidden_dim, condition_dim, 
            depth, heads, dim_head, mlp_dim, 
            sequence_dim=sequence_dim, out_proj=output_proj
        )
        
    def encode(self, x, lengths=None):
        return self.encoder(x, lengths)

    def decode(self, x, c=None, lengths=None):
        return self.decoder(x, c, lengths)

    def forward(self, x, lengths=None):
        encoded = self.encode(x, lengths)
        decoded = self.decode(x, encoded, lengths)
        return decoded, encoded

class JEPA(nn.Module):
    def __init__(
            self,
            pixel_encoder,
            action_encoder,
            predictor
    ):
        super().__init__()

        self.pixel_encoder = pixel_encoder
        self.pixel_projector = MLP(self.pixel_encoder.config.hidden_size, self.pixel_encoder.config.hidden_size*2)

        self.action_encoder = action_encoder
        self.predictor = predictor

        # The pixel encoder is a frozen, pretrained backbone - never updated by the optimizer.
        for param in self.pixel_encoder.parameters():
            param.requires_grad = False
        self.pixel_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.pixel_encoder.eval()  # stay frozen regardless of the rest of the model's mode
        return self

    def encode_pixels(self, pixels):
        """
        pixels: (B, T, C, H, W) - a chain of T observation frames, already resized/cropped/
        normalized by the real HF preprocessor (done in chain_collate_fn, inside the
        DataLoader workers, so this CPU-bound step overlaps with GPU compute instead of
        blocking the forward pass).
        Returns CLS-pooled, per-frame embeddings: (B, T, hidden_size).
        """
        B, T, c, h, w = pixels.shape
        images = pixels.reshape(B * T, c, h, w)

        encoded_pixels = self.pixel_encoder(pixel_values=images).last_hidden_state[:, 0, :]  # CLS token
        encoded_pixels = self.pixel_projector(encoded_pixels)

        return encoded_pixels.reshape(B, T, -1)

    def encode_actions(self, actions, lengths=None):
        return self.action_encoder.encode(actions, lengths)

    def decode_actions(self, actions, conditions, lengths=None):
        return self.action_encoder.decode(actions, conditions, lengths)

    def predict(self, pixels, hop_input, hop_lengths=None):
        """
        pixels: (B, T, H, W, C) - the chain's T observation frames.
        hop_input: (B*(T-1), max_len, action_dim) - the T-1 hops between them, flattened.
        hop_lengths: (B*(T-1),) - each hop's true (START + actions) length.
        """
        B, T = pixels.shape[0], pixels.shape[1]

        obs = self.encode_pixels(pixels)  # (B, T, D)

        hop_emb_flat = self.encode_actions(hop_input, hop_lengths)  # (B*(T-1), 1, D_cond)
        hop_emb = hop_emb_flat.squeeze(1).reshape(B, T - 1, -1)  # (B, T-1, D_cond)

        x, target = obs[:, :-1], obs[:, 1:]  # (B, T-1, D) each
        pred = self.predictor(x, hop_emb)  # per-position AdaLN: hop_emb[:,t] conditions x[:,t]

        decoded_actions = self.decode_actions(hop_input, hop_emb_flat, hop_lengths)

        return {
            "obs": obs,
            "pred": pred,
            "target": target,
            "hop_emb": hop_emb,
            "hop_emb_flat": hop_emb_flat,
            "decoded_actions": decoded_actions,
        }

    def forward(self, pixels, hop_input, hop_lengths=None):
        return self.predict(pixels, hop_input, hop_lengths)
    
    ####################
    ## Inference only ##
    ####################

    def rollout(self, pixels, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        H = info["pixels"].size(2)
        b_orig, s_orig, t_orig, a_orig = action_sequence.shape
        
        B, S, T, A = action_sequence.shape
        
        device = action_sequence.device
        action_sequence = action_sequence.view(B, S, T, -1, 2)
        B, S, T, D, A = action_sequence.shape
        
        start_padding = torch.full((B, S, T, 1, A), -2.0, device=device)
        end_padding = torch.full((B, S, T, 1, A), -3.0, device=device)
        action_sequence = torch.cat([start_padding, action_sequence, end_padding], dim=3)
                
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init["action"] = _init["action"].flatten(0, 1)
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            B, S, T, D = act.shape
            act = act.flatten(0, 1)
            act_emb, _ = self.action_encoder.encode(act)
            act = act.unflatten(0, (B, S))
            act_emb = act_emb.squeeze(1).unflatten(0, (B, S))
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            # act_trunc = act_emb
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_b, act_s, _, _ = act.shape
        act = act.flatten(0, 1)
        act_emb, _ = self.action_encoder.encode(act)  # (BS, T, A_emb)
        act = act.unflatten(0, (act_b, act_s))
        act_emb = act_emb.squeeze(1).unflatten(0, (act_b, act_s))
        
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)
        B, S, D = emb.shape
        
        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=b_orig, s=s_orig)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, predicted_emb, goal_emb):
        """Compute the cost between predicted embeddings and goal embeddings.
        predicted_emb: (B, S, T-1, dim)
        goal_emb: (B, S, T, dim)
        """
        pred_emb = predicted_emb  # (B,S, T-1, dim)
        goal_emb = goal_emb  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, initial_state, goal, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given a goal and initial state."""
        initial_state = self.encode(initial_state)
        goal = self.encode(goal)

        predicted_emb = self.rollout(initial_state, action_candidates)

        cost = self.criterion(predicted_emb, goal)
        
        return cost