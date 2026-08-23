import os
from functools import partial
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from einops import rearrange


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

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        # x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


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

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        c = self.conditional_proj(c)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
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
            nn.Embedding(sequence_dim, hidden_dim)
        )

        for _ in range(depth):
            self.layers.append(
                Block(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )
            

        self.output_proj = (
            nn.Linear(hidden_dim, input_dim)
            if out_proj
            else nn.Identity()
        )

        if out_proj:
            self.mlp = FeedForward(input_dim, mlp_dim, dropout=dropout)
        else:
            self.mlp = FeedForward(hidden_dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, action_dim)
        """
        _, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_enc(torch.arange(seq_len, device=x.device))

        for block in self.layers:
            x = block(x)

        x = self.output_proj(x)

        last = x[:, -1:, :]
        
        return self.mlp(last)
    
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
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
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = nn.LayerNorm(hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
        )
        
        self.pos_enc = (
            nn.Embedding(sequence_dim, hidden_dim)
        )
        
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
        )

        for _ in range(depth):
            self.layers.append(
                ConditionalBlock(hidden_dim, input_dim, heads, dim_head, mlp_dim, dropout=dropout)
            )

        if out_proj:
            self.mlp = FeedForward(input_dim, mlp_dim, dropout=dropout)
        else:
            self.mlp = FeedForward(hidden_dim, mlp_dim, dropout=dropout)

    def forward(self, x, c=None):
        """
        x: (batch, sequence_dim, action_dim)
        c: (batch, 1, embed_dim)
        """
        _, seq_len, _ = x.shape
        
        c = self.mlp(c)
        c = self.input_norm(c)
        
        x = self.input_proj(x)
        x = x + self.pos_enc(torch.arange(seq_len, device=x.device))
        
        for block in self.layers:
            x = block(x, c)
            
        x = self.norm(x)

        x = self.output_proj(x)
            
        return x

class Transpressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
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
            input_dim, hidden_dim, depth, heads, 
            dim_head, mlp_dim, dropout=dropout, 
            sequence_dim=sequence_dim, out_proj=output_proj
        )

        self.decoder = TransformerDecoder(
            input_dim, hidden_dim, output_dim, 
            depth, heads, dim_head, mlp_dim, 
            sequence_dim=sequence_dim, out_proj=output_proj
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x, c=None):
        return self.decoder(x, c)