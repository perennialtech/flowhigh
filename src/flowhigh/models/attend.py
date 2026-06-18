import torch
from einops import rearrange
from torch import einsum, nn
from torch.nn import Module
from torch.nn import functional as F

from .pos_emb import apply_rotary_pos_emb


class Attend(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        flash: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = scale
        self.flash = flash

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, dim_head = q.shape

        if self.scale is not None:
            q = q * (self.scale / (dim_head**-0.5))

        if mask is not None:
            mask = mask.expand(-1, heads, q_len, -1)

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def forward(self, q, k, v, mask=None):
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        if mask is not None and mask.ndim != 4:
            mask = rearrange(mask, "b j -> b 1 1 j")

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum("b h i j, b h j d -> b h i d", attn, v)


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

        self.attend = Attend(
            dropout,
            flash=flash,
            scale=qk_norm_scale if qk_norm else None,
        )

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

        if rotary_emb is not None:
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask=mask)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))
