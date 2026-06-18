from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import normalize_flow_name

FLOW_METHODS = (
    "basic",
    "independent_adaptive",
    "independent_constant",
    "independent_mix",
)

INDEPENDENT_FLOW_METHODS = (
    "independent_adaptive",
    "independent_constant",
    "independent_mix",
)


@dataclass(frozen=True)
class FlowSample:
    x_t: Tensor
    velocity: Tensor
    cutoff_bins: Tensor | None = None


def _time_view(t: Tensor) -> Tensor:
    return t.view(t.shape[0], 1, 1)


def mel_cutoff_bins(mel: Tensor, percentile: float = 0.9995) -> Tensor:
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    energy = torch.exp(mel).abs().sum(dim=1).cumsum(dim=-1)
    threshold = energy[:, -1:] * percentile
    cutoff = torch.searchsorted(energy.contiguous(), threshold.contiguous()).squeeze(-1)
    return cutoff.clamp(0, mel.shape[-1]).long()


def mel_replace(
    samples: Tensor,
    source: Tensor,
    cutoff_bins: Tensor | int | list[int],
) -> Tensor:
    if torch.is_tensor(cutoff_bins):
        cutoff = cutoff_bins.to(device=samples.device, dtype=torch.long)
    else:
        cutoff = torch.as_tensor(cutoff_bins, device=samples.device, dtype=torch.long)

    mel_idx = torch.arange(samples.shape[-1], device=samples.device)
    use_source = mel_idx.view(1, 1, -1) < cutoff.view(-1, 1, 1)
    return torch.where(use_source, source, samples)


@dataclass(frozen=True)
class FlowPath:
    kind: str = "basic"
    sigma: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "kind", normalize_flow_name(self.kind))

    @property
    def is_independent(self) -> bool:
        return self.kind in INDEPENDENT_FLOW_METHODS

    def training_sample(self, x1: Tensor, cond: Tensor, t: Tensor) -> FlowSample:
        if self.kind == "basic":
            eps = torch.randn_like(x1)
            t_ = _time_view(t)
            sigma_t = 1 - (1 - self.sigma) * t_
            return FlowSample(
                x_t=sigma_t * eps + t_ * x1,
                velocity=x1 - (1 - self.sigma) * eps,
            )

        if self.kind == "independent_adaptive":
            eps = torch.randn_like(cond)
            x0 = cond.detach()
            t_ = _time_view(t)
            sigma_t = 1 - (1 - self.sigma) * t_
            return FlowSample(
                x_t=t_ * x1 + (1 - t_) * x0 + sigma_t * eps,
                velocity=(x1 - x0) - (1 - self.sigma) * eps,
            )

        if self.kind == "independent_constant":
            x0 = cond.detach()
            t_ = _time_view(t)
            mu_t = t_ * x1 + (1 - t_) * x0
            noise = 0.0 if self.sigma == 0.0 else self.sigma * torch.randn_like(cond)
            return FlowSample(x_t=mu_t + noise, velocity=x1 - x0)

        if self.kind == "independent_mix":
            eps = torch.randn_like(cond)
            x0 = cond.detach()
            t_ = _time_view(t)
            cutoff = mel_cutoff_bins(cond)

            x_t_high = t_ * x1 + (1 - (1 - self.sigma) * t_) * eps
            x_t_low = t_ * x1 + (1 - t_) * x0 + self.sigma * eps

            velocity_high = x1 - (1 - self.sigma) * eps
            velocity_low = x1 - x0

            return FlowSample(
                x_t=mel_replace(x_t_high, x_t_low, cutoff),
                velocity=mel_replace(velocity_high, velocity_low, cutoff),
                cutoff_bins=cutoff,
            )

        raise ValueError(f"Unsupported flow kind: {self.kind}")

    def initial_state(
        self,
        cond: Tensor,
        *,
        std_1: float | None = None,
        std_2: float | None = None,
        cutoff_bins: Tensor | None = None,
    ) -> Tensor:
        if self.kind == "basic":
            return torch.randn_like(cond)

        std_1 = 1.0 if std_1 is None else std_1
        std_2 = self.sigma if std_2 is None else std_2

        if self.kind in {"independent_adaptive", "independent_constant"}:
            if std_2 == 0.0:
                return cond if std_1 == 1.0 else cond * std_1
            return cond * std_1 + torch.randn_like(cond) * std_2

        if self.kind == "independent_mix":
            eps = torch.randn_like(cond)
            cutoff = mel_cutoff_bins(cond) if cutoff_bins is None else cutoff_bins
            low = cond if std_2 == 0.0 and std_1 == 1.0 else cond * std_1 + eps * std_2
            return mel_replace(eps, low, cutoff)

        raise ValueError(f"Unsupported flow kind: {self.kind}")
