from __future__ import annotations

import torch
from torch import nn

from .mel import MelSpectrogram


class MelCodec(nn.Module):
    def __init__(self, mel: MelSpectrogram, vocoder: nn.Module | None = None):
        super().__init__()
        self.mel = mel
        self.vocoder = vocoder

    @property
    def n_mels(self) -> int:
        return self.mel.n_mels

    @property
    def sampling_rate(self) -> int:
        return self.mel.sampling_rate

    @property
    def n_fft(self) -> int:
        return self.mel.n_fft

    @property
    def win_length(self) -> int:
        return self.mel.win_length

    @property
    def hop_length(self) -> int:
        return self.mel.hop_length

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mel.encode(audio)

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        if self.vocoder is None:
            raise RuntimeError("MelCodec has no vocoder")
        return self.vocoder(mel)
