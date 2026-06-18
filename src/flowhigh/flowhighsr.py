from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchaudio.transforms import Resample

from .audio import (BigVGANVocoder, MelCodec, MelSpectrogram,
                    SpectralLowbandMerge)
from .config import FlowHighConfig
from .flow import FlowMatcher, FlowPath, ODESampler
from .models import FlowHigh
from .pretrained import (REPO_ID, PretrainedBundle, download_pretrained,
                         load_model_weights)


class FlowHighSR(nn.Module):
    def __init__(
        self,
        *,
        matcher: FlowMatcher,
        codec: MelCodec,
        postprocessor: SpectralLowbandMerge | None = None,
    ):
        super().__init__()
        self.matcher = matcher
        self.codec = codec
        self.postprocessor = (
            SpectralLowbandMerge(
                n_fft=codec.n_fft,
                hop_length=codec.hop_length,
                win_length=codec.win_length,
            )
            if postprocessor is None
            else postprocessor
        )
        self._resamplers: dict[tuple[int, int, str], Resample] = {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _resample(
        self, audio: torch.Tensor, source_sr: int, target_sr: int
    ) -> torch.Tensor:
        if source_sr == target_sr:
            return audio

        key = (source_sr, target_sr, str(audio.device))
        resampler = self._resamplers.get(key)

        if resampler is None:
            resampler = Resample(source_sr, target_sr).to(audio.device).eval()
            self._resamplers[key] = resampler

        return resampler(audio)

    @torch.inference_mode()
    def enhance(
        self,
        audio: torch.Tensor | np.ndarray,
        *,
        sample_rate: int,
        steps: int = 1,
        guidance_scale: float = 1.0,
        output_sample_rate: int | None = None,
    ) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = np.asarray(audio).squeeze()

            if audio.ndim != 1:
                raise ValueError("numpy audio must be mono or squeezable to mono")

            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)

            audio = torch.from_numpy(audio)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim != 2:
            raise ValueError("audio must have shape [T] or [B, T]")

        audio = audio.to(self.device, dtype=torch.float32, non_blocking=True)
        audio = audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

        model_sr = self.codec.sampling_rate
        source = self._resample(audio, sample_rate, model_sr)

        condition = self.codec.encode(source)
        predicted_mel = self.matcher.sample(
            condition,
            steps=steps,
            guidance_scale=guidance_scale,
        )

        generated = self.codec.decode(predicted_mel).squeeze(1)
        generated = self.postprocessor(generated, source, length=source.size(-1))

        if output_sample_rate is not None and output_sample_rate != model_sr:
            generated = self._resample(generated, model_sr, output_sample_rate)

        return generated

    @classmethod
    def from_local(
        cls,
        ckpt_dir: str | Path,
        *,
        device: str | torch.device,
        compile_model: bool = False,
    ) -> "FlowHighSR":
        bundle = PretrainedBundle(Path(ckpt_dir))
        config = FlowHighConfig.from_json(bundle.config_path)

        mel = MelSpectrogram(
            n_mels=config.audio.n_mels,
            sampling_rate=config.audio.sampling_rate,
            f_max=config.audio.f_max,
            f_min=config.audio.f_min,
            n_fft=config.audio.n_fft,
            win_length=config.audio.win_length,
            hop_length=config.audio.hop_length,
        )

        vocoder = BigVGANVocoder(
            config_path=bundle.vocoder_config_path,
            checkpoint_path=bundle.vocoder_path,
        )

        codec = MelCodec(mel, vocoder)

        model = FlowHigh(
            dim_in=config.network.dim_in,
            depth=config.network.depth,
            dim=config.network.dim,
            dim_head=config.network.dim_head,
            heads=config.network.heads,
            architecture=config.network.architecture,
            attn_flash=config.network.attn_flash,
        )

        load_model_weights(model, bundle.model_path)

        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")

        matcher = FlowMatcher(
            model,
            flow=FlowPath(config.flow.kind, sigma=config.flow.sigma),
            sampler=ODESampler(
                method=config.flow.ode_method,
                backend=config.flow.ode_backend,
            ),
        )

        instance = cls(matcher=matcher, codec=codec).to(device).eval()
        return instance

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device,
        repo_id: str = REPO_ID,
        compile_model: bool = False,
    ) -> "FlowHighSR":
        bundle = download_pretrained(repo_id)
        return cls.from_local(
            bundle.root,
            device=device,
            compile_model=compile_model,
        )
