from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .losses import combine_masks, lengths_to_mask, masked_mse
from .paths import FlowPath, mel_cutoff_bins, mel_replace
from .sampler import ODESampler


class FlowMatcher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        flow: FlowPath | None = None,
        sampler: ODESampler | None = None,
        cond_drop_prob: float = 0.0,
        inference_autocast_dtype: torch.dtype | None = torch.float16,
    ):
        super().__init__()
        self.model = model
        self.flow = FlowPath() if flow is None else flow
        self.sampler = ODESampler() if sampler is None else sampler
        self.cond_drop_prob = cond_drop_prob
        self.inference_autocast_dtype = inference_autocast_dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _velocity(
        self,
        x: Tensor,
        *,
        t: Tensor,
        condition: Tensor,
        mask: Tensor | None,
        guidance_scale: float,
    ) -> Tensor:
        if self.inference_autocast_dtype is None or x.device.type != "cuda":
            logits = self.model(
                x,
                times=t,
                cond=condition,
                self_attn_mask=mask,
                cond_drop_prob=0.0,
            )

            if guidance_scale == 1.0:
                return logits.to(dtype=x.dtype)

            null_logits = self.model(
                x,
                times=t,
                cond=condition,
                self_attn_mask=mask,
                cond_drop_prob=1.0,
            )
            return (null_logits + (logits - null_logits) * guidance_scale).to(
                dtype=x.dtype
            )

        with torch.autocast("cuda", dtype=self.inference_autocast_dtype):
            logits = self.model(
                x,
                times=t,
                cond=condition,
                self_attn_mask=mask,
                cond_drop_prob=0.0,
            )

            if guidance_scale == 1.0:
                return logits.to(dtype=x.dtype)

            null_logits = self.model(
                x,
                times=t,
                cond=condition,
                self_attn_mask=mask,
                cond_drop_prob=1.0,
            )
            return (null_logits + (logits - null_logits) * guidance_scale).to(
                dtype=x.dtype
            )

    @torch.inference_mode()
    def sample(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        steps: int = 4,
        guidance_scale: float = 1.0,
        std_1: float | None = None,
        std_2: float | None = None,
        replace_low_band: bool = False,
    ) -> Tensor:
        if condition.ndim != 3:
            raise ValueError("condition must have shape [B, T, C]")

        condition = condition.to(self.device, dtype=torch.float32, non_blocking=True)
        mask = (
            None
            if mask is None
            else mask.to(self.device, dtype=torch.bool, non_blocking=True)
        )

        cutoff_bins = (
            mel_cutoff_bins(condition)
            if self.flow.kind == "independent_mix" or replace_low_band
            else None
        )

        if self.flow.is_independent:
            std_1 = 1.0 if std_1 is None else std_1
            std_2 = self.flow.sigma if std_2 is None else std_2

        y0 = self.flow.initial_state(
            condition,
            std_1=std_1,
            std_2=std_2,
            cutoff_bins=cutoff_bins,
        )

        self.model.eval()

        def fn(t: Tensor, x: Tensor) -> Tensor:
            return self._velocity(
                x,
                t=t,
                condition=condition,
                mask=mask,
                guidance_scale=guidance_scale,
            )

        sampled = self.sampler.sample(y0, steps, fn)

        if replace_low_band:
            cutoff_bins = (
                mel_cutoff_bins(condition) if cutoff_bins is None else cutoff_bins
            )
            sampled = mel_replace(sampled, condition, cutoff_bins)

        return sampled

    def loss(
        self,
        target: Tensor,
        condition: Tensor,
        *,
        lengths: Tensor | None = None,
        mask: Tensor | None = None,
        cond_freq_masking: bool = False,
        weighted_loss: bool = False,
    ) -> Tensor:
        if target.ndim != 3 or condition.ndim != 3:
            raise ValueError("target and condition must have shape [B, T, C]")

        target = target.to(self.device, dtype=torch.float32, non_blocking=True)
        condition = condition.to(self.device, dtype=torch.float32, non_blocking=True)

        max_frames = max(target.shape[1], condition.shape[1])

        if target.shape[1] < max_frames:
            target = F.pad(target, (0, 0, 0, max_frames - target.shape[1]))

        if condition.shape[1] < max_frames:
            condition = F.pad(condition, (0, 0, 0, max_frames - condition.shape[1]))

        lengths = (
            None
            if lengths is None
            else lengths.to(self.device, dtype=torch.long, non_blocking=True)
        )
        length_mask = (
            lengths_to_mask(lengths, max_frames) if lengths is not None else None
        )

        mask = (
            None
            if mask is None
            else mask.to(self.device, dtype=torch.bool, non_blocking=True)
        )
        attn_mask = combine_masks(mask, length_mask)

        batch = target.shape[0]
        times = torch.rand(batch, device=self.device, dtype=target.dtype)
        sample = self.flow.training_sample(target, condition, times)

        pred = self.model(
            sample.x_t,
            times=times,
            cond=condition,
            self_attn_mask=attn_mask,
            cond_drop_prob=self.cond_drop_prob,
            cond_freq_masking=cond_freq_masking,
        )

        weight = None
        if weighted_loss:
            cutoff = sample.cutoff_bins
            cutoff = mel_cutoff_bins(condition) if cutoff is None else cutoff
            mel_idx = torch.arange(pred.shape[-1], device=pred.device)
            high = mel_idx.unsqueeze(0) >= cutoff.unsqueeze(1)
            weight = torch.where(
                high,
                torch.full((batch, pred.shape[-1]), 2.0, device=pred.device),
                torch.ones((batch, pred.shape[-1]), device=pred.device),
            ).to(dtype=pred.dtype)

        return masked_mse(pred, sample.velocity, attn_mask, weight=weight)

    def forward(self, target: Tensor, condition: Tensor, **kwargs) -> Tensor:
        return self.loss(target, condition, **kwargs)
