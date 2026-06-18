from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import cheby1, resample_poly, sosfiltfilt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def _peak_normalize(wave: torch.Tensor) -> torch.Tensor:
    peak = wave.abs().amax().clamp_min(1e-8)
    return wave / peak


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant")

    if len(audio) > length:
        return audio[:length]

    return audio


class AudioFolder(Dataset):
    def __init__(
        self,
        folder: str | Path,
        *,
        sample_rate: int,
        audio_extension: str = ".wav",
    ):
        super().__init__()

        self.folder = Path(folder)
        self.sample_rate = sample_rate

        if not self.folder.exists():
            raise FileNotFoundError(self.folder)

        self.files = sorted(self.folder.glob(f"**/*{audio_extension}"))

        if not self.files:
            raise RuntimeError(f"no {audio_extension} files found in {self.folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio, sr = torchaudio.load(self.files[index])
        audio = audio.mean(dim=0)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return _peak_normalize(audio.float())


class RandomLowpassResample:
    def __init__(
        self,
        *,
        sample_rate: int,
        min_sr: int = 4000,
        max_sr: int = 32000,
        method: str = "scipy",
    ):
        self.sample_rate = sample_rate
        self.min_sr = min_sr
        self.max_sr = max_sr
        self.method = method

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        random_sr = random.randrange(self.min_sr, self.max_sr + 1000, 1000)
        wave = audio.detach().cpu().numpy()

        nyquist = self.sample_rate / 2
        highcut = min(random_sr / 2, nyquist * 0.99)

        order = random.randint(1, 11)
        ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
        sos = cheby1(order, ripple, highcut / nyquist, btype="lowpass", output="sos")
        filtered = sosfiltfilt(sos, wave)

        if self.method == "scipy":
            down = resample_poly(filtered, random_sr, self.sample_rate)
            up = resample_poly(down, self.sample_rate, random_sr)
        elif self.method == "librosa":
            import librosa

            down = librosa.resample(
                filtered,
                orig_sr=self.sample_rate,
                target_sr=random_sr,
                res_type="soxr_hq",
            )
            up = librosa.resample(
                down,
                orig_sr=random_sr,
                target_sr=self.sample_rate,
                res_type="soxr_hq",
            )
        else:
            raise ValueError(f"Unsupported degradation method: {self.method}")

        return torch.from_numpy(_fit_length(up, len(wave)).copy()).float()


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        source: AudioFolder,
        degradation: RandomLowpassResample,
    ):
        super().__init__()
        self.source = source
        self.degradation = degradation

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index: int):
        target = self.source[index]
        condition = self.degradation(target)
        return target, condition, target.shape[-1]


def collate_audio(batch):
    targets, conditions, lengths = zip(*batch)
    return (
        pad_sequence(targets, batch_first=True),
        pad_sequence(conditions, batch_first=True),
        torch.tensor(lengths, dtype=torch.long),
    )


def get_dataloader(dataset: Dataset, **kwargs):
    kwargs.setdefault("num_workers", min(8, os.cpu_count() or 1))

    if kwargs["num_workers"] > 0:
        kwargs.setdefault("persistent_workers", True)

    return DataLoader(dataset, collate_fn=collate_audio, **kwargs)
