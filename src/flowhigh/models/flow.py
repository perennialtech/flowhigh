from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

from .convnext import ConvNeXtBlock
from .pos_emb import LearnedSinusoidalPosEmb
from .transformer import ConvPositionEmbed, Transformer


def prob_mask_like(shape, prob: float, device):
    if prob <= 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)

    if prob >= 1:
        return torch.ones(shape, device=device, dtype=torch.bool)

    return torch.rand(shape, device=device) < prob


def mask_frequency(cond: torch.Tensor) -> torch.Tensor:
    batch, _, mel_dim = cond.shape

    if mel_dim <= 32:
        return cond

    cond = cond.clone()

    for i in range(batch):
        height = int(torch.randint(10, min(21, mel_dim), (), device=cond.device).item())
        start_max = max(1, mel_dim - height - 20)
        start = 20 + int(torch.randint(0, start_max, (), device=cond.device).item())
        cond[i, :, start : start + height] = cond.amin() + 1e-3

    return cond


class FlowHigh(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int = 256,
        dim: int = 1024,
        depth: int = 24,
        dim_head: int = 64,
        heads: int = 16,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        time_hidden_dim: int | None = None,
        conv_pos_embed_kernel_size: int = 31,
        conv_pos_embed_groups: int | None = None,
        attn_dropout: float = 0.0,
        attn_flash: bool = True,
        attn_qk_norm: bool = True,
        architecture: str = "transformer",
    ):
        super().__init__()

        if architecture not in {"transformer", "convnext"}:
            raise ValueError("architecture must be 'transformer' or 'convnext'")

        self.architecture = architecture
        time_hidden_dim = dim if time_hidden_dim is None else time_hidden_dim

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU(),
        )

        self.to_embed = nn.Linear(dim_in * 2, dim)
        self.register_buffer("null_cond", torch.zeros(dim_in), persistent=True)
        self.conv_embed = ConvPositionEmbed(
            dim=dim,
            kernel_size=conv_pos_embed_kernel_size,
            groups=conv_pos_embed_groups,
        )

        if architecture == "transformer":
            self.transformer = Transformer(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                ff_dropout=ff_dropout,
                attn_dropout=attn_dropout,
                attn_flash=attn_flash,
                attn_qk_norm=attn_qk_norm,
                adaptive_rmsnorm=True,
                adaptive_rmsnorm_cond_dim_in=time_hidden_dim,
            )
        else:
            self.convnext = nn.ModuleList(
                [
                    ConvNeXtBlock(
                        dim=dim,
                        intermediate_dim=dim * 3,
                        layer_scale_init_value=1,
                        hidden_dim=time_hidden_dim,
                    )
                    for _ in range(depth)
                ]
            )
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.to_pred = nn.Linear(dim, dim_in, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x: torch.Tensor,
        *,
        times: torch.Tensor,
        cond: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
        cond_freq_masking: bool = False,
    ) -> torch.Tensor:
        attn_mask = self_attn_mask if self_attn_mask is not None else mask

        if x.shape != cond.shape:
            raise ValueError(
                f"x and cond must have the same shape, got {x.shape} and {cond.shape}"
            )

        batch, _, cond_dim = cond.shape

        if cond_dim != self.null_cond.shape[0]:
            raise ValueError(
                f"expected mel dimension {self.null_cond.shape[0]}, got {cond_dim}"
            )

        if times.ndim == 0:
            times = repeat(times, "-> b", b=batch)
        elif times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, "1 -> b", b=batch)

        if cond_freq_masking and self.training:
            cond = mask_frequency(cond)

        if cond_drop_prob > 0:
            drop = prob_mask_like((batch,), cond_drop_prob, cond.device)
            null_cond = self.null_cond.to(device=cond.device, dtype=cond.dtype)
            cond = torch.where(rearrange(drop, "b -> b 1 1"), null_cond, cond)

        x = self.to_embed(torch.cat((x, cond), dim=-1))
        x = self.conv_embed(x, mask=attn_mask) + x

        time_emb = self.sinu_pos_emb(times)

        if self.architecture == "transformer":
            x = self.transformer(x, mask=attn_mask, adaptive_rmsnorm_cond=time_emb)
        else:
            x = x.transpose(1, 2)

            for block in self.convnext:
                x = block(x, cond=time_emb)

            x = self.final_layer_norm(x.transpose(1, 2))

        return self.to_pred(x)
