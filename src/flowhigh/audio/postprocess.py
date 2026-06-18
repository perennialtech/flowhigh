from __future__ import annotations

import torch
from torch import nn
from torchaudio.transforms import InverseSpectrogram, Spectrogram


class SpectralLowbandMerge(nn.Module):
    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        threshold: float = 0.99,
    ):
        super().__init__()
        self.threshold = threshold
        self.stft = Spectrogram(
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None,
            pad_mode="constant",
        )
        self.istft = InverseSpectrogram(
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            pad_mode="constant",
        )

    def cutoff_bins(self, spec: torch.Tensor) -> torch.Tensor:
        energy = spec.abs().sum(dim=-1).cumsum(dim=-1)
        cutoff = torch.searchsorted(
            energy.contiguous(),
            (energy[:, -1:] * self.threshold).contiguous(),
        ).squeeze(-1)
        return cutoff.clamp(0, spec.size(1)).long()

    def forward(
        self,
        generated: torch.Tensor,
        source: torch.Tensor,
        *,
        length: int,
    ) -> torch.Tensor:
        if generated.ndim != 2 or source.ndim != 2:
            raise ValueError("generated and source must have shape [B, T]")

        gen_spec = self.stft(generated)
        src_spec = self.stft(source)

        cutoff = self.cutoff_bins(src_spec)
        frames = min(gen_spec.size(-1), src_spec.size(-1))

        gen_spec = gen_spec[:, :, :frames]
        src_spec = src_spec[:, :, :frames]

        freq_idx = torch.arange(gen_spec.size(1), device=gen_spec.device)
        use_source = freq_idx.view(1, -1, 1) < cutoff.view(-1, 1, 1)
        merged = torch.where(use_source, src_spec, gen_spec)

        audio = self.istft(merged, length=length)
        peak = audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return audio / peak * 0.99
