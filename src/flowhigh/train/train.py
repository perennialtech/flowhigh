import sys

import numpy as np
import torch

from ..audio import MelCodec, MelSpectrogram
from ..config import FlowHighConfig
from ..flow import FlowMatcher, FlowPath, ODESampler
from ..models import FlowHigh
from .data import AudioFolder, RandomLowpassResample, SuperResolutionDataset
from .trainer import FlowHighTrainer


def main(config_path: str = "configs/config.json"):
    if not torch.cuda.is_available():
        raise RuntimeError("CPU training is not supported")

    config = FlowHighConfig.from_json(config_path)

    torch.manual_seed(config.train.random_seed)
    np.random.seed(config.train.random_seed)

    source = AudioFolder(
        config.train.data_path,
        sample_rate=config.audio.sampling_rate,
        audio_extension=config.train.audio_extension,
    )
    degradation = RandomLowpassResample(
        sample_rate=config.audio.sampling_rate,
        min_sr=config.train.downsample_min,
        max_sr=config.train.downsample_max,
        method=config.train.downsampling_method,
    )
    dataset = SuperResolutionDataset(source, degradation)

    codec = MelCodec(
        MelSpectrogram(
            n_mels=config.audio.n_mels,
            sampling_rate=config.audio.sampling_rate,
            f_max=config.audio.f_max,
            f_min=config.audio.f_min,
            n_fft=config.audio.n_fft,
            win_length=config.audio.win_length,
            hop_length=config.audio.hop_length,
        )
    )

    model = FlowHigh(
        dim_in=config.network.dim_in,
        dim=config.network.dim,
        depth=config.network.depth,
        dim_head=config.network.dim_head,
        heads=config.network.heads,
        architecture=config.network.architecture,
        attn_flash=config.network.attn_flash,
    )

    matcher = FlowMatcher(
        model,
        flow=FlowPath(config.flow.kind, sigma=config.flow.sigma),
        sampler=ODESampler(
            method=config.flow.ode_method,
            backend=config.flow.ode_backend,
        ),
    )

    trainer = FlowHighTrainer(
        matcher=matcher,
        codec=codec,
        dataset=dataset,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.json")
