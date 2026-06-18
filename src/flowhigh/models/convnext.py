from typing import Optional

import torch
from torch import nn
from torch.nn import Module


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(hidden_dim, embedding_dim)
        self.shift = nn.Linear(hidden_dim, embedding_dim)

        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond)
        shift = self.shift(cond)

        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        scale, shift = map(lambda t: t.unsqueeze(1).expand_as(x), (scale, shift))
        return x * scale + shift


class ConvNeXtBlock(Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = hidden_dim is not None
        self.norm = (
            AdaLayerNorm(dim, hidden_dim, eps=1e-6)
            if hidden_dim is not None
            else nn.LayerNorm(dim, eps=1e-6)
        )
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        x = self.dwconv(x).transpose(1, 2)

        if self.adanorm:
            if cond is None:
                raise ValueError("cond is required for adaptive layer norm")
            x = self.norm(x, cond)
        else:
            x = self.norm(x)

        x = self.pwconv2(self.act(self.pwconv1(x)))

        if self.gamma is not None:
            x = self.gamma * x

        return residual + x.transpose(1, 2)
