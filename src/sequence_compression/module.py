import os
from functools import partial
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from einops import rearrange

from transformers import AutoImageProcessor, AutoModel, ViTConfig, ViTModel

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

# vit_hf, taken from stable_pretraining
def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Vision Transformer using HuggingFace transformers.

    This provides a clean, well-maintained ViT implementation with native support for:
    - Masking via bool_masked_pos parameter
    - Learnable mask token
    - Easy access to CLS and patch tokens

    Args:
        size: Model size - "tiny", "small", "base", or "large"
        patch_size: Patch size (default: 16)
        image_size: Input image size (default: 224)
        pretrained: Load pretrained weights from HuggingFace Hub
        use_mask_token: Whether to include learnable mask token (needed for iBOT)
        **kwargs: Additional ViTConfig parameters

    Returns:
        HuggingFace ViTModel

    Example:
        >>> backbone = vit_hf("tiny", use_mask_token=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>>
        >>> # Without masking
        >>> output = backbone(x)
        >>> cls_token = output.last_hidden_state[:, 0, :]
        >>> patch_tokens = output.last_hidden_state[:, 1:, :]
        >>>
        >>> # With masking (for iBOT student)
        >>> masks = torch.zeros(2, 196, dtype=torch.bool)
        >>> masks[:, :59] = True  # Mask 30%
        >>> output = backbone(x, bool_masked_pos=masks)
    """

    # ViT size configurations (matching timm/DINOv3)
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
        "huge": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
        },
    }

    if size not in size_configs:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(size_configs.keys())}"
        )

    config_params = size_configs[size]
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        # Try to load pretrained model from HF Hub
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        # logging.info(f"Loading pretrained ViT from {model_name}")
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        # logging.info(f"Created ViT-{size} from scratch with config: {config_params}")

    # IMPORTANT: Set model to always interpolate position encodings for dynamic input sizes
    # This allows processing images of different sizes (e.g., 224x224 global + 96x96 local views)
    # Must be set as instance attribute, not in config
    model.config.interpolate_pos_encoding = True

    return model

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
        norm_fn: type[nn.LayerNorm | nn.BatchNorm1d]=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity() # type: ignore
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
            if input_dim != hidden_dim else
            nn.Identity()
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.transformer = TransformerDecoder(
            input_dim, hidden_dim, condition_dim, depth, heads, dim_head, mlp_dim,
            dropout=dropout, sequence_dim=sequence_dim, out_proj=out_proj
        )

    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self, x, c=None):
        return self.transformer(x, c)

class NaiveActionEmbedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def encode(self, x, lengths=None):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x

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
            sequence_dim=sequence_dim, out_proj=output_proj, dropout=dropout
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

        # Choosing 80 here because of some analysis in `representation_analysis.ipynb`. It seems for my 
        # data, the amount of informative eigenvectors is roughly 80. The idea here is to apply a learned 
        # projection from R^385 to R^80, which we would then feed into the encoder
        self.subspace_size = 80
        self.pixel_projector = MLP(
            self.pixel_encoder.config.hidden_size, 
            self.pixel_encoder.config.hidden_size*2,
            norm_fn=nn.BatchNorm1d
        )

        self.action_encoder = action_encoder
        self.predictor: ARPredictor = predictor

        self.predictor_projector = MLP(
            self.predictor.get_input_dim(),
            self.predictor.get_input_dim()*2,
            norm_fn=nn.BatchNorm1d
        )

    def train(self, mode=True):
        super().train(mode)
        self.pixel_encoder.eval()  # stay frozen regardless of the rest of the model's mode
        return self

    def encode_pixels(self, pixels):
        """
        pixels: (B, T, C, H, W) - a chain of T observation frames, already resized/cropped/
        rescaled/normalized by the real HF preprocessor (channel-first, as it produces them).
        Returns CLS-pooled, per-frame embeddings: (B, T, hidden_size).
        """
        batch, timestep, c, h, w = pixels.shape

        images = pixels.reshape(batch * timestep, c, h, w)

        encoded_pixels = self.pixel_encoder(pixel_values=images).last_hidden_state[:, 0, :]  # CLS token
        encoded_pixels = self.pixel_projector(encoded_pixels)

        return encoded_pixels.reshape(batch, timestep, -1)

    def encode_actions(self, actions, lengths=None):
        return self.action_encoder.encode(actions, lengths)

    def decode_actions(self, actions, conditions, lengths=None):
        return self.action_encoder.decode(actions, conditions, lengths)

    def predict(self, pixels, hop_input, hop_lengths=None):
        """
        pixels: (B, T, C, H, W) - the chain's T observation frames.
        hop_input: (B*(T-1), max_len, action_dim) - the T-1 hops between them, flattened.
        hop_lengths: (B*(T-1),) - each hop's true (START + actions) length.
        """
        B, T = pixels.shape[0], pixels.shape[1]

        obs = self.encode_pixels(pixels)  # (B, T, D)

        hop_emb = self.encode_actions(hop_input, hop_lengths)  # (B*T, 1, D_cond)
        hop_emb = hop_emb.squeeze(1).reshape(B, T, -1)
        hop_emb = hop_emb[:, :-1]

        x, target = obs[:, :-1], obs[:, 1:]  # (B, T-1, D) each
        pred = self.predictor(x, hop_emb)  # per-position AdaLN: hop_emb[:,t] conditions x[:,t]

        B, T, D = pred.shape
        pred = pred.reshape(B*T, D)
        pred = self.pixel_projector(pred)
        pred = pred.reshape(B, T, D)

        return {
            "obs": obs,
            "pred": pred,
            "target": target,
            "hop_emb": hop_emb,
        }

    def forward(self, pixels, hop_input, hop_lengths=None):
        return self.predict(pixels, hop_input, hop_lengths)
    
    ####################
    ## Inference only ##
    ####################

    def rollout(self, pixels, latent_actions, real_hop_input=None, real_hop_lengths=None, history_size=None):
        """Imagine a trajectory forward from real observed history using CEM-sampled latent actions.
        pixels: (B, T_hist, C, H, W) - the real observed history. Identical across every one
            of the S candidates, so it's encoded once here rather than pre-broadcast across S.
        latent_actions: (B, S, T_plan, D_cond) - per-candidate action embeddings sampled
            directly from N(0, I) by CEM; fed straight to the predictor, bypassing the action
            encoder entirely (the action encoder's sigreg training term is what makes raw
            Gaussian draws valid inputs here).
        real_hop_input: (B*(T_hist-1), max_len, action_dim), optional - the real hops
            connecting the history frames. Omit when T_hist == 1 (no action history yet).
        real_hop_lengths: (B*(T_hist-1),), optional - each real hop's true length.
        history_size: int, optional - max context length kept for the predictor at each
            step (defaults to keeping the whole growing history).
        Returns: (B, S, D) - the predicted embedding after the full T_plan-step plan.
        """
        B, T_hist = pixels.shape[:2]
        S, T_plan, D_cond = latent_actions.shape[1:]

        # encode the real history once - it's identical across all S candidates
        obs = self.encode_pixels(pixels)  # (B, T_hist, D)
        if T_hist > 1:
            hop_emb = self.encode_actions(real_hop_input, real_hop_lengths)  # (B*(T_hist-1), 1, D_cond)
            hop_emb = hop_emb.squeeze(1).unflatten(0, (B, T_hist - 1))  # (B, T_hist-1, D_cond)
        else:
            hop_emb = obs.new_zeros(B, 0, D_cond)

        # broadcast the real history across S candidates, then flatten (B, S) -> BS
        obs = obs.unsqueeze(1).expand(B, S, T_hist, -1).reshape(B * S, T_hist, -1)
        hop_emb = hop_emb.unsqueeze(1).expand(B, S, T_hist - 1, D_cond).reshape(B * S, T_hist - 1, D_cond)
        latent_actions = latent_actions.flatten(0, 1)  # (BS, T_plan, D_cond)

        for t in range(T_plan):
            hop_emb = torch.cat([hop_emb, latent_actions[:, t : t + 1]], dim=1)  # (BS, len+1, D_cond)
            ctx_obs = obs if history_size is None else obs[:, -history_size:]
            ctx_hop = hop_emb if history_size is None else hop_emb[:, -history_size:]
            pred = self.predictor(ctx_obs, ctx_hop)[:, -1:]  # (BS, 1, D)
            obs = torch.cat([obs, pred], dim=1)

        return obs[:, -1].unflatten(0, (B, S))  # (B, S, D)

    def cost(self, predicted_emb, goal_emb):
        """Compute the MSE cost between predicted and goal embeddings.
        predicted_emb: (B, S, dim) - predicted next-observation embedding per candidate.
        goal_emb: (B, dim) - target goal embedding.
        """
        # goal_emb = goal_emb.unsqueeze(1).expand_as(predicted_emb)  # (B, S, dim)

        cost = F.mse_loss(
            predicted_emb,
            goal_emb.detach(),
            reduction="none",
        ).sum(dim=-1)  # (B, S)

        # cost = 1 - F.cosine_similarity(
        #     predicted_emb,
        #     goal_emb.detach(),
        #     dim=-1,
        # )  # (B, S)

        return cost

  
    def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
        return self.get_cost(info_dict, action_candidates)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of CEM-sampled latent action plans given a goal and real history.
        info: dict - contains the real observed history and goal.
        action_candidates: (B, S, T_plan, D_cond) - CEM-sampled per-candidate action embeddings.
        """
        pixels = info_dict["pixels"]  # (B, T_hist, C, H, W)
        goal = info_dict["goal"]  # (B, C, H, W)

        # For compatibility with stable_worldmodel
        if len(pixels.shape) > 5:
            B, S, _, _, C, H, W = pixels.shape
            pixels = pixels.reshape(B*S, 1, C, H, W)
            goal = goal.reshape(B*S, 1, C, H, W)
        else:
            B, S, C, H, W = pixels.shape

        action_embs = self.encode_actions(action_candidates.flatten(0, 1))  # (BS, 1, D_cond)
        action_embs = action_embs.reshape(B, S, 1, -1)

        predicted_emb = self.rollout(
            pixels[0].unsqueeze(0), action_embs
        )  # (B, S, D)

        goal_emb = self.encode_pixels(goal[0].unsqueeze(0)) # (B, D)

        predicted_emb = predicted_emb.reshape(B, S, -1)
        goal_emb = goal_emb.tile((1, 300, 1))

        cost = self.cost(predicted_emb, goal_emb)

        return cost