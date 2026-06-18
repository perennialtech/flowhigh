from __future__ import annotations

import torch
from torch import Tensor


def lengths_to_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    lengths = lengths.long()
    if max_length is None:
        max_length = int(lengths.max().item())

    steps = torch.arange(max_length, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    valid_masks = [mask.bool() for mask in masks if mask is not None]
    if not valid_masks:
        return None

    out = valid_masks[0]
    for mask in valid_masks[1:]:
        out = out & mask
    return out


def masked_mse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    weight: Tensor | None = None,
) -> Tensor:
    loss = (pred - target).square()

    if weight is not None:
        if weight.ndim == 2:
            weight = weight[:, None, :]
        loss = loss * weight.to(device=loss.device, dtype=loss.dtype)

    loss = loss.mean(dim=-1)

    if mask is None:
        return loss.mean()

    mask = mask.to(device=loss.device, dtype=torch.bool)
    loss = loss.masked_fill(~mask, 0.0)

    num = loss.sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1)
    return (num / den).mean()
