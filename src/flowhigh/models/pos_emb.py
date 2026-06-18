import math
from typing import Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Module


class LearnedSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("dim must be divisible by 2")

        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1)


class RotaryEmbedding(Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    def forward(self, t: Union[int, Tensor]):
        if isinstance(t, int):
            t = torch.arange(t, device=self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()
