import logging
from functools import partial
from pathlib import Path
from typing import Any

from beartype import beartype

from einops import rearrange, repeat
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torchode as to
from torchdiffeq import odeint
from torchaudio.functional import resample

from .models.modules import exists
from .models.common import unpack_one, pack_one
from .models import FLowHigh
from .utils import sequence_mask

LOGGER = logging.getLogger(__file__)
logging.basicConfig(filename="model_debug.log", level=logging.INFO)

CFM_METHODS = (
    "basic_cfm",
    "independent_cfm_adaptive",
    "independent_cfm_constant",
    "independent_cfm_mix",
)

INDEPENDENT_CFM_METHODS = (
    "independent_cfm_adaptive",
    "independent_cfm_constant",
    "independent_cfm_mix",
)

DEFAULT_COMPILE_BUCKET_FRAMES = (
    *range(16, 256 + 1, 16),
    *range(320, 4096 + 1, 64),
)

DEFAULT_WARMUP_FRAME_COUNTS = (100, 250, 500, 1000)


# mel helpers
mel_basis = {}
hann_window = {}


def interpolate_1d(t, length, mode="bilinear"):
    "pytorch does not offer interpolation 1d, so hack by converting to 2d"

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, "b n -> b 1 n")

    t = rearrange(t, "b d n -> b d n 1")
    t = F.interpolate(t, (length, 1), mode=mode)
    t = rearrange(t, "b d n 1 -> b d n")

    if implicit_one_channel:
        t = rearrange(t, "b 1 n -> b n")

    t = t.to(dtype)

    return t


def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value=0.0)

    return t


# mask construction helpers
def mask_from_start_end_indices(seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device=device, dtype=torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()  # start
    mask &= seq < end[..., None].long()

    return mask


def mask_from_frac_lengths(seq_len: int, frac_lengths: Tensor):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device=device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))


class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma=0.0,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        use_torchode=False,
        cfm_method="basic_cfm",
        torchdiffeq_ode_method="midpoint",  # [euler, midpoint]
        torchode_method_klass=to.Tsit5,
        cond_drop_prob=0.0,
        inference_autocast_dtype: torch.dtype | None = None,
        compile_flowhigh: bool = False,
        compile_bucket_frames: int | tuple[int, ...] = DEFAULT_COMPILE_BUCKET_FRAMES,
    ):
        super().__init__()
        self.sigma = sigma
        self.flowhigh = flowhigh
        self.cond_drop_prob = cond_drop_prob
        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass
        self.cfm_method = cfm_method
        self.inference_autocast_dtype = inference_autocast_dtype
        self.compile_flowhigh = compile_flowhigh
        if isinstance(compile_bucket_frames, int):
            self.compile_bucket_frames = (
                (compile_bucket_frames,) if compile_bucket_frames > 0 else ()
            )
        else:
            self.compile_bucket_frames = tuple(
                sorted({bucket for bucket in compile_bucket_frames if bucket > 0})
            )
        self.__dict__["_compiled_flowhigh"] = None
        self.odeint_kwargs: dict[str, Any]
        self.odeint_kwargs = dict(
            atol=ode_atol, rtol=ode_rtol, method=torchdiffeq_ode_method
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict=True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")
        self.load_state_dict(pkg["model"], strict=strict)
        return pkg

    def _flowhigh_for_inference(self):
        if not self.compile_flowhigh or self.device.type != "cuda":
            return self.flowhigh

        if not hasattr(torch, "compile"):
            return self.flowhigh

        compiled = self.__dict__.get("_compiled_flowhigh")
        if compiled is not None:
            return compiled

        try:
            compiled = torch.compile(self.flowhigh, mode="reduce-overhead")
        except Exception:
            self.compile_flowhigh = False
            return self.flowhigh

        self.__dict__["_compiled_flowhigh"] = compiled
        return compiled

    def _bucket_frames(self, frame_count: int) -> int:
        if not self.compile_flowhigh or not self.compile_bucket_frames:
            return frame_count

        for bucket in self.compile_bucket_frames:
            if frame_count <= bucket:
                return bucket

        return frame_count

    def _forward_with_cond_scale(
        self,
        model,
        x,
        *,
        times,
        cond,
        cond_scale,
        cond_mask,
        self_attn_mask,
    ):
        autocast_enabled = (
            self.inference_autocast_dtype is not None and x.device.type == "cuda"
        )
        autocast_dtype = self.inference_autocast_dtype or torch.float16

        with torch.autocast(
            device_type=x.device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(
                x,
                times=times,
                cond=cond,
                cond_drop_prob=0.0,
                cond_mask=cond_mask,
                self_attn_mask=self_attn_mask,
            )

            if cond_scale == 1.0:
                return logits.to(dtype=x.dtype)

            null_logits = model(
                x,
                times=times,
                cond=cond,
                cond_drop_prob=1.0,
                cond_mask=cond_mask,
                self_attn_mask=self_attn_mask,
            )

            out = null_logits + (logits - null_logits) * cond_scale
            return out.to(dtype=x.dtype)

    def _fixed_step_solve(self, y0, time_steps, ode_fn, method):
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than 0")

        y = y0
        dt = 1.0 / time_steps

        for step in range(time_steps):
            t0 = y.new_tensor(step * dt)

            if method == "euler":
                y = y + dt * ode_fn(t0, y)
                continue

            if method == "midpoint":
                tm = y.new_tensor((step + 0.5) * dt)
                k1 = ode_fn(t0, y)
                y = y + dt * ode_fn(tm, y + 0.5 * dt * k1)
                continue

            raise ValueError(f"Unsupported fixed-step ODE method: {method}")

        return y

    # For mel repalcement
    def locate_cutoff_freq(self, mel, percentile=0.9995):
        cutoff = self.mel_cutoff_bins(mel.unsqueeze(0), percentile=percentile)
        return int(cutoff[0].item())

    def mel_replace_ops(self, samples, input, cutoff_melbins):
        if torch.is_tensor(cutoff_melbins):
            cutoff_melbins = cutoff_melbins.to(
                device=samples.device,
                dtype=torch.long,
            )
        else:
            cutoff_melbins = torch.tensor(
                cutoff_melbins,
                device=samples.device,
                dtype=torch.long,
            )

        mel_idx = torch.arange(samples.shape[-1], device=samples.device)
        use_input = mel_idx.view(1, 1, -1) < cutoff_melbins.view(-1, 1, 1)
        return torch.where(use_input, input, samples), cutoff_melbins

    def mel_cutoff_bins(self, mel, percentile=0.9995):
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)

        energy = torch.exp(mel).abs().sum(dim=1).cumsum(dim=-1)
        threshold = energy[:, -1:] * percentile
        cutoff = torch.searchsorted(
            energy.contiguous(),
            threshold.contiguous(),
        ).squeeze(-1)
        return cutoff.clamp(0, mel.shape[-1]).to(torch.long)

    @torch.inference_mode()
    def warmup(
        self,
        *,
        batch_size: int = 1,
        frame_counts: tuple[int, ...] | None = None,
        time_steps: int = 1,
        cond_scale: float = 1.0,
        cfm_method: str | None = None,
    ):
        if self.device.type != "cuda" or not self.compile_flowhigh:
            return

        if frame_counts is None:
            frame_counts = DEFAULT_WARMUP_FRAME_COUNTS

        frame_counts = tuple(int(frames) for frames in frame_counts if int(frames) > 0)
        if not frame_counts:
            return

        self.eval()
        dim = self.flowhigh.null_cond.shape[0]

        for frames in frame_counts:
            cond = torch.zeros(
                batch_size,
                frames,
                dim,
                device=self.device,
                dtype=torch.float32,
            )
            cond_mask = torch.ones(
                batch_size,
                frames,
                device=self.device,
                dtype=torch.bool,
            )
            self.sample(
                cond=cond,
                cond_mask=cond_mask,
                time_steps=time_steps,
                cond_scale=cond_scale,
                decode_to_audio=False,
                cfm_method=cfm_method,
            )

        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def sample(
        self,
        *,
        cond: Tensor | None = None,
        cond_mask=None,
        time_steps=4,
        cond_scale=1.0,
        decode_to_audio=True,
        std_1: float | None = None,
        std_2: float | None = None,
        mel_pp=False,
        cfm_method: str | None = None,
    ) -> Tensor:
        if cond is None:
            raise ValueError("`cond` is required for sampling")

        cond = cond.to(self.device, dtype=torch.float32, non_blocking=True)

        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device, dtype=torch.bool, non_blocking=True)

        if cfm_method not in CFM_METHODS:
            cfm_method = self.cfm_method

        if cfm_method not in CFM_METHODS:
            raise ValueError(f"Unsupported CFM method: {cfm_method}")

        if cfm_method in INDEPENDENT_CFM_METHODS:
            std_1 = 1.0 if std_1 is None else std_1
            std_2 = self.sigma if std_2 is None else std_2

        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            audio_enc_dec = self.flowhigh.audio_enc_dec
            assert audio_enc_dec is not None

            audio_enc_dec.eval()
            cond = audio_enc_dec.encode(cond)

        original_cond = cond
        original_cond_frames = cond.shape[1]
        needs_cutoff_bins = cfm_method == "independent_cfm_mix" or mel_pp
        cutoff_bins = self.mel_cutoff_bins(cond) if needs_cutoff_bins else None

        if self.compile_flowhigh and self.compile_bucket_frames:
            target_frames = self._bucket_frames(cond.shape[1])
            pad_frames = target_frames - cond.shape[1]

            if pad_frames:
                if cond_mask is None:
                    cond_mask = torch.ones(
                        (cond.shape[0], cond.shape[1]),
                        device=cond.device,
                        dtype=torch.bool,
                    )

                cond = F.pad(cond, (0, 0, 0, pad_frames), value=0.0)
                cond_mask = F.pad(cond_mask, (0, pad_frames), value=False)

        self_attn_mask = cond_mask
        shape = cond.shape  # [B, Time, Channel]
        batch = shape[0]

        # neural ode
        self.flowhigh.eval()
        inference_model = self._flowhigh_for_inference()

        # ode function
        def ode_fn(t, x, *, packed_shape=None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, "b *")

            out = self._forward_with_cond_scale(
                inference_model,
                x,
                times=t,
                cond=cond,
                cond_scale=cond_scale,
                cond_mask=cond_mask,
                self_attn_mask=self_attn_mask,
            )

            if exists(packed_shape):
                out = rearrange(out, "b ... -> b (...)")
            return out  # out.shape : [1, Time, mel_channel]

        if cfm_method == "basic_cfm":
            y0 = torch.randn_like(cond)

        elif cfm_method in {"independent_cfm_adaptive", "independent_cfm_constant"}:
            if std_2 == 0.0:
                y0 = cond if std_1 == 1.0 else cond * std_1
            else:
                epsilon = torch.randn_like(cond)
                y0 = cond * std_1 + epsilon * std_2

        elif cfm_method == "independent_cfm_mix":
            # y0 from intended prior
            assert cutoff_bins is not None
            epsilon = torch.randn_like(cond)
            if std_2 == 0.0:
                y0_low = cond if std_1 == 1.0 else cond * std_1
            else:
                y0_low = cond * std_1 + epsilon * std_2
            y0_high = epsilon
            y0, _ = self.mel_replace_ops(y0_high, y0_low, cutoff_bins)
        else:
            raise AssertionError("unreachable CFM method branch")

        t = torch.linspace(0, 1, time_steps + 1, device=self.device)
        ode_method = self.odeint_kwargs["method"]

        if not self.use_torchode and ode_method in {"euler", "midpoint"}:
            LOGGER.debug("sampling with manual fixed-step %s", ode_method)
            sampled = self._fixed_step_solve(y0, time_steps, ode_fn, ode_method)

        elif not self.use_torchode:

            LOGGER.debug("sampling with torchdiffeq")
            trajectory = odeint(ode_fn, y0, t, **self.odeint_kwargs)  # bottle neck
            sampled = trajectory[-1]

            # # trajectory plot
            # n = len(trajectory)
            # for i in range(n):
            #     plt.figure(figsize=(12, 4))
            #     plt.imshow(numpy.rot90(trajectory[i].squeeze().cpu().numpy(), 1), aspect='auto', origin='upper', interpolation='none')
            #     plt.colorbar()
            #     plt.title(f'trajectory[{i}]')
            #     plt.xlabel('X-axis')
            #     plt.ylabel('Y-axis')

            #     plt.savefig(f'__trajectory[{i}].png', dpi=300, bbox_inches='tight')
            #     plt.close()

        else:
            LOGGER.debug("sampling with torchode")
            t = repeat(t, "n -> b n", b=batch)
            y0, packed_shape = pack_one(y0, "b *")
            fn = partial(ode_fn, packed_shape=packed_shape)
            term = to.ODETerm(fn)  # pyright: ignore[reportArgumentType]
            step_method = self.torchode_method_klass(term=term)
            step_size_controller = to.IntegralController(
                atol=self.odeint_kwargs["atol"],
                rtol=self.odeint_kwargs["rtol"],
                term=term,
            )
            solver = to.AutoDiffAdjoint(
                step_method, step_size_controller
            )  # pyright: ignore[reportArgumentType]
            init_value = to.InitialValueProblem(
                y0=y0, t_eval=t
            )  # pyright: ignore[reportArgumentType]
            sol = solver.solve(
                init_value
            )  # pyright: ignore[reportFunctionMemberAccess]
            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, "b *")

        if sampled.shape[1] != original_cond_frames:
            sampled = sampled[:, :original_cond_frames]
            cond = original_cond

        if mel_pp:
            assert cutoff_bins is not None
            sampled, cutoff_bins = self.mel_replace_ops(sampled, cond, cutoff_bins)

        if not decode_to_audio or not exists(self.flowhigh.audio_enc_dec):
            return sampled

        audio_enc_dec = self.flowhigh.audio_enc_dec
        assert audio_enc_dec is not None
        return audio_enc_dec.decode(sampled)

    # this is for training only.
    def forward(
        self,
        x1: Tensor,
        *,
        mask=None,
        cond: Tensor | None = None,
        cond_mask=None,
        cond_lengths: Tensor | None = None,
        input_sampling_rate: int | None = None,
        cond_freq_masking=False,
        random_sr=None,
        weighted_loss=None,
        cfm_method: str | None = None,  # not necessary
    ):
        if cond is None:
            raise ValueError("`cond` is required for training")

        if cfm_method not in [
            "basic_cfm",
            "independent_cfm_adaptive",
            "independent_cfm_constant",
            "independent_cfm_mix",
        ]:
            cfm_method = self.cfm_method

        batch, seq_len, dtype, sigma_min = *x1.shape[:2], x1.dtype, self.sigma
        input_is_raw_audio, cond_is_raw_audio = map(
            is_probably_audio_from_shape, (x1, cond)
        )

        if any([input_is_raw_audio, cond_is_raw_audio]):
            audio_enc_dec = self.flowhigh.audio_enc_dec
            assert (
                audio_enc_dec is not None
            ), "audio_enc_dec must be set on FLowHigh to train directly on raw audio"
            audio_enc_dec_sampling_rate = audio_enc_dec.sampling_rate
            if input_sampling_rate is None:
                input_sampling_rate = audio_enc_dec_sampling_rate

            with torch.no_grad():
                audio_enc_dec.eval()
                # Making Ground truth mel-spectrogram
                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = audio_enc_dec.encode(x1)  # x1.shape : [B, Time, channel]

                # Making mel-spectrogram which are empty in high-freqeuncy information
                if exists(cond) and cond_is_raw_audio:
                    cond = resample(
                        cond, input_sampling_rate, audio_enc_dec_sampling_rate
                    )
                cond = audio_enc_dec.encode(cond)  # cond.shape : [B, Time, channel]

        if x1.size(1) != cond.size(1):
            max_timelength = max(x1.size(1), cond.size(1))
            x1 = F.pad(x1, (0, 0, max_timelength - x1.size(1), 0))
            cond = F.pad(cond, (0, 0, max_timelength - cond.size(1), 0))

        # main conditional flow logic is below
        times = torch.rand((batch,), dtype=dtype, device=self.device)
        t = rearrange(times, "b -> b 1 1")
        cutoff_bins: list[int] | None = None

        if cfm_method == "basic_cfm":
            """
            probability path: N(t x1, 1 - (1 - sigma) t)
            mu_t: t * x1
            sigma_t: 1 - (1 - sigma_min)t
            sigma_min = 1e-4

            sample x_t: sigma_t * x0 + t * x1 = (1 - (1 - sigma_min) * t) * x0 + t * x1
            target vector field: u_t = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t) = x1 - (1 - sigma_min) * x0

            if sigma_min = 0, then basic_cfm same with rectified-flow from standard normal distribution N(0,I)
            """
            # x0 is gaussian noise
            x0 = torch.randn_like(x1)  # [B, Time, channel]

            sigma_t = 1 - (1 - sigma_min) * t

            # sample xt = noisy speech (\psi_t (x_0|x_1))
            w = sigma_t * x0 + t * x1  # [B, Time, channel]
            # w = (1 - (1 - sigma_min) * t) * x0 + t * x1  # [B, Time, channel]

            # target vector field u_t
            flow = x1 - (1 - sigma_min) * x0  # [B, Time, channel]
            # flow = (x1 - (1 - sigma_min) * w) / (1- (1 - sigma_min) *t)  # [B, Time, channel]

        elif cfm_method == "independent_cfm_adaptive":
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, 1 - (1 - sigma_min) t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t:  1 - (1 - sigma_min) t

            sample x_t: mean + sigma * eps = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = { (x1-x0) - (1-sigma_min)(xt-x0) } / { 1 - (1 - sigma_min) t } = (x1-x0) - (1-sigma_min) * epsilon

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """

            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach()

            mu_t = t * x1 + (1 - t) * x0
            sigma_t = 1 - (1 - sigma_min) * t

            # sample xt
            w = mu_t + sigma_t * epsilon

            # target vector field u_t
            flow = (x1 - x0) - (
                1 - sigma_min
            ) * epsilon  # { (x1-x0) - (1-sigma_min)*(w-x0) } / {1 - (1 - sigma_min)*t} # [B, Time, channel]

        elif cfm_method == "independent_cfm_constant":
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, sigma_t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t: sigma_min (small enough)

            sample x_t: mean + sigma*eps(eps~N(0,I)) = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = x1 - x0

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach()

            mu_t = t * x1 + (1 - t) * x0

            # sample xt
            if sigma_min == 0:
                w = mu_t
            else:
                epsilon = torch.randn_like(cond)
                w = mu_t + sigma_min * epsilon

            # target vector field u_t
            flow = x1 - x0  # [B, Time, channel]

        elif cfm_method == "independent_cfm_mix":
            """
            q(z) = q(x0)q(x1)
            probability path_high: N(    t * x1          , 1 - (1 - sigma) t)
            probability path_low : N(t * x1 + (1 - t) *x0,       sigma_min    )

            x0: x^mel_low

            sample x_t:
            target vector field: u_t =
            """

            # # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)

            # get cutoff mel bins of LR mel
            cutoff_bins = self.mel_cutoff_bins(cond)

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach()

            # sample xt_high
            mu_t_high = t * x1
            sigma_t_high = 1 - (1 - sigma_min) * t
            xt_high = mu_t_high + sigma_t_high * epsilon  # [B, Time, channel]

            # sample xt_low
            mu_t_low = t * x1 + (1 - t) * x0
            sigma_t_low = sigma_min
            xt_low = mu_t_low + sigma_t_low * epsilon

            w, _ = self.mel_replace_ops(xt_high, xt_low, cutoff_bins)

            # target vector field u_t
            flow_high = x1 - (1 - sigma_min) * epsilon
            flow_low = x1 - x0  # [B, Time, channel]
            flow, _ = self.mel_replace_ops(flow_high, flow_low, cutoff_bins)
        else:
            raise AssertionError("unreachable CFM method branch")

        # x1.shape = cond.shape = x0.shape = w.shape = flow.shape = [Batch, Time, mel_bin]

        # Training mode!
        self.flowhigh.train()

        # Cut a small segment of mel-spectrogram
        if cond_lengths is None:
            raise ValueError("`cond_lengths` is required for training")
        audio_enc_dec = self.flowhigh.audio_enc_dec
        assert audio_enc_dec is not None

        cond_lengths = cond_lengths.to(torch.int32)
        max_cond_lengths = x1.size(1)
        x_mask = sequence_mask(cond_lengths, max_cond_lengths).unsqueeze(1)
        out_size = 2 * audio_enc_dec.sampling_rate // audio_enc_dec.hop_length

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (cond_lengths - out_size).clamp(0)

            import random

            out_offset = torch.tensor(
                [
                    random.choice(range(0, end)) if end > 0 else 0
                    for end in max_offset.cpu().tolist()
                ],
                dtype=torch.long,
                device=cond_lengths.device,
            )

            w_cut = torch.zeros(
                w.shape[0],
                out_size,
                audio_enc_dec.n_mels,
                dtype=w.dtype,
                device=w.device,
            )
            flow_cut = torch.zeros(
                flow.shape[0],
                out_size,
                audio_enc_dec.n_mels,
                dtype=flow.dtype,
                device=flow.device,
            )
            cond_cut = torch.zeros(
                cond.shape[0],
                out_size,
                audio_enc_dec.n_mels,
                dtype=cond.dtype,
                device=cond.device,
            )

            x_cut_lengths: list[Tensor] = []

            for i in range(w.shape[0]):
                w_, flow_, cond_, out_offset_ = w[i], flow[i], cond[i], out_offset[i]

                # w_.shape = flow_.shape = cond_.shape = [Time, channel]
                x_cut_length = out_size + (cond_lengths[i] - out_size).clamp(None, 0)
                x_cut_lengths.append(x_cut_length)

                cut_lower, cut_upper = out_offset_, out_offset_ + x_cut_length
                w_cut[i, :x_cut_length, :] = w_[cut_lower:cut_upper, :]
                flow_cut[i, :x_cut_length, :] = flow_[cut_lower:cut_upper, :]
                cond_cut[i, :x_cut_length, :] = cond_[cut_lower:cut_upper, :]

            x_cut_lengths = torch.stack(x_cut_lengths).to(cond_lengths.device)
            x_cut_mask = sequence_mask(x_cut_lengths).unsqueeze(1).to(x_mask)

            w = w_cut
            flow = flow_cut
            cond = cond_cut
            x_mask = x_cut_mask

        # forward
        loss = self.flowhigh(
            x=w,
            cond=cond,
            cond_mask=cond_mask,
            times=times,
            target=flow,
            self_attn_mask=mask,
            cond_drop_prob=self.cond_drop_prob,
            cond_freq_masking=cond_freq_masking,
            random_sr=random_sr,
            weighted_loss=weighted_loss,
            cutoff_bins=cutoff_bins,
        )
        return loss
