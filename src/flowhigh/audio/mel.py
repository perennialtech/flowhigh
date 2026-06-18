from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def spectral_normalize(magnitudes: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(magnitudes, min=1e-5))


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = 256,
        sampling_rate: int = 48000,
        f_max: int = 24000,
        f_min: int = 20,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 480,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.f_max = f_max
        self.f_min = f_min
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        from librosa.filters import mel as librosa_mel_fn

        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )

        self.register_buffer(
            "mel_basis",
            torch.from_numpy(mel).float(),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )

    @property
    def latent_dim(self) -> int:
        return self.n_mels

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError("audio must have shape [B, T]")

        pad = int((self.n_fft - self.hop_length) / 2)
        mode = "reflect" if audio.shape[-1] > pad else "constant"
        audio = F.pad(audio.unsqueeze(1), (pad, pad), mode=mode).squeeze(1)

        window = self.window.to(device=audio.device, dtype=audio.dtype)
        mel_basis = self.mel_basis.to(device=audio.device, dtype=audio.dtype)

        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = spec.abs().clamp_min(1e-9)
        mel = torch.matmul(mel_basis, spec)
        mel = spectral_normalize(mel)
        return rearrange(mel, "b c t -> b t c")
