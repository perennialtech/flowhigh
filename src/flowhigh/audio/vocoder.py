from __future__ import annotations

import json
from pathlib import Path

import torch
from einops import rearrange
from torch import nn


class BigVGANVocoder(nn.Module):
    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
    ):
        super().__init__()

        from bigvgan.bigvgan import BigVGAN
        from bigvgan.env import AttrDict

        with open(config_path) as f:
            config = AttrDict(json.load(f))

        self.vocoder = BigVGAN(config, use_cuda_kernel=torch.cuda.is_available())

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.vocoder.load_state_dict(checkpoint["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

        for param in self.vocoder.parameters():
            param.requires_grad = False

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [B, T, C]")

        return self.vocoder(rearrange(mel, "b t c -> b c t"))
