from pathlib import Path

import scipy
import numpy as np
import librosa
import torch
import torchode
from huggingface_hub import hf_hub_download
from torchaudio.transforms import Resample

from .models import FLowHigh, MelVoco
from .cfm_superresolution import (
    ConditionalFlowMatcherWrapper,
    DEFAULT_COMPILE_BUCKET_FRAMES,
    DEFAULT_WARMUP_FRAME_COUNTS,
)
from .postprocessing import PostProcessing

REPO_ID = "ResembleAI/FlowHigh"


class FlowHighSR(ConditionalFlowMatcherWrapper):
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma=0.0,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        use_torchode=False,
        cfm_method="basic_cfm",
        torchdiffeq_ode_method="midpoint",  # [euler, midpoint]
        torchode_method_klass=torchode.Tsit5,
        cond_drop_prob=0.0,
        inference_autocast_dtype: torch.dtype | None = torch.float16,
        compile_flowhigh=True,
        compile_bucket_frames: int | tuple[int, ...] = DEFAULT_COMPILE_BUCKET_FRAMES,
        #
        upsampling_method="scipy",
    ):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        super().__init__(
            flowhigh=flowhigh,
            sigma=sigma,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
            use_torchode=use_torchode,
            cfm_method=cfm_method,
            torchdiffeq_ode_method=torchdiffeq_ode_method,
            torchode_method_klass=torchode_method_klass,
            cond_drop_prob=cond_drop_prob,
            inference_autocast_dtype=inference_autocast_dtype,
            compile_flowhigh=compile_flowhigh,
            compile_bucket_frames=compile_bucket_frames,
        )
        self.upsampling_method = upsampling_method
        self.postproc = PostProcessing()
        self._resamplers: dict[tuple[int, int, str], Resample] = {}

    @torch.no_grad()
    def _generate_from_condition(self, cond: torch.Tensor, timestep: int):
        cond = cond.to(self.device, dtype=torch.float32, non_blocking=True)
        cond = cond / cond.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.cfm_method == "independent_cfm_adaptive":
            HR_audio = self.sample(
                cond=cond, time_steps=timestep, cfm_method=self.cfm_method, std_2=1.0
            )
        elif self.cfm_method in (
            "basic_cfm",
            "independent_cfm_constant",
            "independent_cfm_mix",
        ):
            HR_audio = self.sample(
                cond=cond, time_steps=timestep, cfm_method=self.cfm_method
            )
        else:
            raise ValueError(f"Unsupported CFM method: {self.cfm_method}")

        HR_audio = HR_audio.squeeze(1)  # [B, T]
        return self.postproc.post_processing(HR_audio, cond, cond.size(-1))

    def _resample_to_target(
        self,
        audio: torch.Tensor,
        sr: int,
        target_sampling_rate: int,
    ):
        if sr == target_sampling_rate:
            return audio

        key = (sr, target_sampling_rate, str(audio.device))
        resampler = self._resamplers.get(key)

        if resampler is None:
            resampler = Resample(sr, target_sampling_rate).to(audio.device).eval()
            self._resamplers[key] = resampler

        return resampler(audio)

    @torch.inference_mode()
    def warmup(
        self,
        *,
        batch_size: int = 1,
        frame_counts: tuple[int, ...] | None = None,
        time_steps: int = 1,
        cond_scale: float = 1.0,
        cfm_method: str | None = None,
        target_sampling_rate: int | None = None,
    ):
        if self.device.type != "cuda":
            return

        if frame_counts is None:
            frame_counts = DEFAULT_WARMUP_FRAME_COUNTS

        frame_counts = tuple(int(frames) for frames in frame_counts if int(frames) > 0)
        if not frame_counts:
            return

        self.eval()
        super().warmup(
            batch_size=batch_size,
            frame_counts=frame_counts,
            time_steps=time_steps,
            cond_scale=cond_scale,
            cfm_method=cfm_method,
        )

        audio_enc_dec = self.flowhigh.audio_enc_dec
        if audio_enc_dec is None:
            torch.cuda.synchronize(self.device)
            return

        target_sampling_rate = (
            audio_enc_dec.sampling_rate
            if target_sampling_rate is None
            else target_sampling_rate
        )

        previous_cfm_method = self.cfm_method
        if cfm_method is not None:
            self.cfm_method = cfm_method

        try:
            for frames in frame_counts:
                audio = torch.zeros(
                    batch_size,
                    frames * audio_enc_dec.hop_length,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.generate_tensor(
                    audio,
                    sr=target_sampling_rate,
                    target_sampling_rate=target_sampling_rate,
                    timestep=time_steps,
                )
        finally:
            self.cfm_method = previous_cfm_method

        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def generate_tensor(
        self,
        audio: torch.Tensor,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
    ):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim != 2:
            raise ValueError("audio must have shape [T] or [B, T]")

        audio = audio.to(self.device, dtype=torch.float32, non_blocking=True)
        cond = self._resample_to_target(audio, sr, target_sampling_rate)
        return self._generate_from_condition(cond, timestep)

    @torch.no_grad()
    def generate_numpy(
        self,
        audio: np.ndarray,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
    ):
        audio = np.asarray(audio).squeeze()

        if audio.ndim != 1:
            raise ValueError("audio must be mono or squeezable to mono")

        if np.max(np.abs(audio)) > 1:
            audio = audio / 32768.0

        # Up sampling the input audio (in Numpy)
        if self.upsampling_method == "scipy":
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = scipy.signal.resample_poly(audio, target_sampling_rate, sr)

        elif self.upsampling_method == "librosa":
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = librosa.resample(
                audio, orig_sr=sr, target_sr=target_sampling_rate, res_type="soxr_hq"
            )
        else:
            raise ValueError(f"Unsupported upsampling method: {self.upsampling_method}")

        cond = torch.as_tensor(
            cond,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return self._generate_from_condition(cond, timestep)

    def set_cfm_method(self, cfm_method):
        self.cfm_method = cfm_method
        # torchdiffeq_ode_method
        # sigma

    @classmethod
    def from_local(cls, ckpt_dir: Path, device) -> "FlowHighSR":
        ckpt_dir = Path(ckpt_dir)
        device = torch.device(device)

        voc = MelVoco(
            vocoder_config=str(ckpt_dir / "bigvgan_48khz_256band.json"),
            vocoder_path=str(ckpt_dir / "bigvgan_48khz_256band.pt"),
        )

        SR_generator = (
            FLowHigh(
                dim_in=voc.n_mels,
                audio_enc_dec=voc,
                depth=2,  # args.n_layers,
                attn_flash=True,
            )
            .to(device)
            .eval()
        )

        cfm_wrapper = cls(
            flowhigh=SR_generator,
            compile_flowhigh=True,
            # cfm_method = args.cfm_method,
            # torchdiffeq_ode_method=args.ode_method,
            # sigma = args.sigma,
        )
        # checkpoint load
        model_checkpoint = torch.load(
            ckpt_dir / "FLowHigh_basic_400k.pt", map_location=device
        )
        cfm_wrapper.load_state_dict(
            model_checkpoint["model"]
        )  # dict_keys(['model', 'optim', 'scheduler'])
        cfm_wrapper = cfm_wrapper.to(device).eval()
        cfm_wrapper.warmup()
        return cfm_wrapper

    @classmethod
    def from_pretrained(cls, device) -> "FlowHighSR":
        for fpath in [
            "FLowHigh_basic_400k.json",
            "bigvgan_48khz_256band.json",
            "FLowHigh_basic_400k.pt",
            "bigvgan_48khz_256band.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)
