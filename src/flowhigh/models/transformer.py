from functools import partial

import torch
from einops import rearrange
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

from .attend import Attention
from .pos_emb import RotaryEmbedding


class ConvPositionEmbed(Module):
    def __init__(self, dim, *, kernel_size, groups=None):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        groups = dim if groups is None else groups
        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        out = rearrange(
            self.dw_conv1d(rearrange(x, "b n c -> b c n")), "b c n -> b n c"
        )

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class AdaptiveRMSNorm(Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        cond_dim = dim if cond_dim is None else cond_dim
        self.scale = dim**0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim=-1) * self.scale
        gamma = rearrange(self.to_gamma(cond), "b d -> b 1 d")
        beta = rearrange(self.to_beta(cond), "b d -> b 1 d")
        return normed * gamma + beta


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )


class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        attn_flash: bool = False,
        adaptive_rmsnorm: bool = False,
        adaptive_rmsnorm_cond_dim_in=None,
        attn_qk_norm: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.rotary_emb = RotaryEmbedding(dim=dim_head)

        rmsnorm_klass = (
            partial(AdaptiveRMSNorm, cond_dim=adaptive_rmsnorm_cond_dim_in)
            if adaptive_rmsnorm
            else RMSNorm
        )

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        None,
                        None,
                        rmsnorm_klass(dim=dim),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=attn_flash,
                            qk_norm=attn_qk_norm,
                        ),
                        rmsnorm_klass(dim=dim),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, mask=None, adaptive_rmsnorm_cond=None):
        rotary_emb = self.rotary_emb(x.shape[1])

        rmsnorm_kwargs = {}
        if adaptive_rmsnorm_cond is not None:
            rmsnorm_kwargs = {"cond": adaptive_rmsnorm_cond}

        for _, _, attn_prenorm, attn, ff_prenorm, ff in self.layers:
            x = (
                attn(
                    attn_prenorm(x, **rmsnorm_kwargs), mask=mask, rotary_emb=rotary_emb
                )
                + x
            )
            x = ff(ff_prenorm(x, **rmsnorm_kwargs)) + x

        return self.final_norm(x)
