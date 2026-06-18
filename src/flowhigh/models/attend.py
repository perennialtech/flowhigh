from functools import wraps

# from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange

from .pos_emb import apply_rotary_pos_emb
from .modules import exists

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig",
    [
        "enable_flash",
        "enable_math",
        "enable_mem_efficient",
    ],
)


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(
        self, dropout: float = 0.0, flash: bool = False, scale: float | None = None
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.scale = scale

        self.flash = flash

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, dim_head = q.shape

        # if scale is given, divide by the default scale that sdpa uses

        if exists(self.scale):
            q = q * (self.scale / (dim_head**-0.5))

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if mask is not None:
            mask = mask.expand(-1, heads, q_len, -1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        if mask is not None and mask.ndim != 4:
            mask = rearrange(mask, "b j -> b 1 1 j")

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out


# attention


class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout: float = 0.0,
        flash: bool = False,
        qk_norm: bool = False,
        qk_norm_scale: float = 10,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash=flash, scale=scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads=heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads=heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, mask=None, rotary_emb=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
