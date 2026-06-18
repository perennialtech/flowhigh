from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO_ID = "ResembleAI/FlowHigh"

PRETRAINED_FILES = (
    "FLowHigh_basic_400k.json",
    "bigvgan_48khz_256band.json",
    "FLowHigh_basic_400k.pt",
    "bigvgan_48khz_256band.pt",
)


@dataclass(frozen=True)
class PretrainedBundle:
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "FLowHigh_basic_400k.json"

    @property
    def model_path(self) -> Path:
        return self.root / "FLowHigh_basic_400k.pt"

    @property
    def vocoder_config_path(self) -> Path:
        return self.root / "bigvgan_48khz_256band.json"

    @property
    def vocoder_path(self) -> Path:
        return self.root / "bigvgan_48khz_256band.pt"


def download_pretrained(repo_id: str = REPO_ID) -> PretrainedBundle:
    from huggingface_hub import hf_hub_download

    last_path: str | None = None
    for filename in PRETRAINED_FILES:
        last_path = hf_hub_download(repo_id=repo_id, filename=filename)

    if last_path is None:
        raise RuntimeError("failed to download pretrained files")

    return PretrainedBundle(Path(last_path).parent)


def _strip_prefix(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]

    if key.startswith("model."):
        return key[len("model.") :]

    if key.startswith("flowhigh."):
        return key[len("flowhigh.") :]

    return key


def load_model_weights(model: nn.Module, path: str | Path) -> None:
    package = torch.load(str(path), map_location="cpu")
    state = package.get("model", package)

    expected = model.state_dict()
    filtered = {}

    for key, value in state.items():
        key = _strip_prefix(key)

        if key.startswith("audio_enc_dec.") or key.startswith("vocoder."):
            continue

        if key in expected and expected[key].shape == value.shape:
            filtered[key] = value

    if not filtered:
        raise RuntimeError(f"no compatible FlowHigh weights found in {path}")

    missing, _ = model.load_state_dict(filtered, strict=False)
    missing = [key for key in missing if key != "null_cond"]

    if missing:
        raise RuntimeError(f"checkpoint is missing model weights: {missing[:10]}")
