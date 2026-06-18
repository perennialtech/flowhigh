"""
FlowHigh public API.

Typical audio super-resolution usage:

    import torch
    from flowhigh import FlowHighSR

    sr = FlowHighSR.from_pretrained(device="cuda")
    enhanced = sr.enhance(audio, sample_rate=16000, steps=1)

Mel-level usage:

    from flowhigh import FlowHigh, FlowMatcher
    from flowhigh.flow import FlowPath, ODESampler

    model = FlowHigh(dim_in=256)
    matcher = FlowMatcher(
        model,
        flow=FlowPath("basic", sigma=0.0),
        sampler=ODESampler(method="midpoint"),
    )

    loss = matcher.loss(target_mel, condition_mel, lengths=lengths)
    generated_mel = matcher.sample(condition_mel, steps=4)
"""

from .audio import BigVGANVocoder, MelCodec, MelSpectrogram
from .flow import FlowMatcher, FlowPath, ODESampler
from .flowhighsr import FlowHighSR
from .models import FlowHigh

__all__ = [
    "BigVGANVocoder",
    "FlowHigh",
    "FlowHighSR",
    "FlowMatcher",
    "FlowPath",
    "MelCodec",
    "MelSpectrogram",
    "ODESampler",
]
