from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FLOW_NAME_ALIASES = {
    "basic": "basic",
    "basic_cfm": "basic",
    "independent_adaptive": "independent_adaptive",
    "independent_cfm_adaptive": "independent_adaptive",
    "independent_constant": "independent_constant",
    "independent_cfm_constant": "independent_constant",
    "independent_mix": "independent_mix",
    "independent_cfm_mix": "independent_mix",
}


def _pick(data: dict[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def normalize_flow_name(name: str) -> str:
    try:
        return FLOW_NAME_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported flow kind: {name}") from exc


@dataclass(frozen=True)
class AudioConfig:
    n_mels: int = 256
    sampling_rate: int = 48000
    f_max: int = 24000
    f_min: int = 20
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 480

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "AudioConfig":
        if not cfg:
            return cls()

        data = _section(cfg, "data") or cfg
        return cls(
            n_mels=int(_pick(data, ("n_mels", "n_mel_channels"), cls.n_mels)),
            sampling_rate=int(
                _pick(data, ("sampling_rate", "samplingrate"), cls.sampling_rate)
            ),
            f_max=int(_pick(data, ("f_max", "mel_fmax"), cls.f_max)),
            f_min=int(_pick(data, ("f_min", "mel_fmin"), cls.f_min)),
            n_fft=int(_pick(data, ("n_fft",), cls.n_fft)),
            win_length=int(_pick(data, ("win_length",), cls.win_length)),
            hop_length=int(_pick(data, ("hop_length",), cls.hop_length)),
        )


@dataclass(frozen=True)
class NetworkConfig:
    dim_in: int = 256
    dim: int = 1024
    depth: int = 2
    dim_head: int = 64
    heads: int = 16
    architecture: str = "transformer"
    attn_flash: bool = True

    @classmethod
    def from_mapping(
        cls, cfg: dict[str, Any] | None, *, dim_in: int = 256
    ) -> "NetworkConfig":
        if not cfg:
            return cls(dim_in=dim_in)

        data = _section(cfg, "model") or cfg
        return cls(
            dim_in=int(_pick(data, ("dim_in", "n_mel_channels"), dim_in)),
            dim=int(_pick(data, ("dim",), cls.dim)),
            depth=int(_pick(data, ("depth", "n_layers"), cls.depth)),
            dim_head=int(_pick(data, ("dim_head",), cls.dim_head)),
            heads=int(_pick(data, ("heads", "n_heads"), cls.heads)),
            architecture=str(_pick(data, ("architecture",), cls.architecture)),
            attn_flash=bool(_pick(data, ("attn_flash",), cls.attn_flash)),
        )


@dataclass(frozen=True)
class FlowConfig:
    kind: str = "basic"
    sigma: float = 0.0
    ode_method: str = "midpoint"
    ode_backend: str = "fixed"
    cond_scale: float = 1.0
    steps: int = 4

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "FlowConfig":
        if not cfg:
            return cls()

        data = (
            _section(cfg, "flow")
            or _section(cfg, "sampling")
            or _section(cfg, "model")
            or cfg
        )

        return cls(
            kind=normalize_flow_name(
                str(_pick(data, ("kind", "cfm_method", "cfm_path"), cls.kind))
            ),
            sigma=float(_pick(data, ("sigma",), cls.sigma)),
            ode_method=str(
                _pick(data, ("ode_method", "torchdiffeq_ode_method"), cls.ode_method)
            ),
            ode_backend=str(_pick(data, ("ode_backend",), cls.ode_backend)),
            cond_scale=float(_pick(data, ("cond_scale",), cls.cond_scale)),
            steps=int(_pick(data, ("steps", "time_steps", "timestep"), cls.steps)),
        )


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = ""
    audio_extension: str = ".wav"
    downsampling_method: str = "scipy"
    downsample_min: int = 4000
    downsample_max: int = 32000
    batch_size: int = 1
    steps: int = 400_000
    warmup_steps: int = 0
    lr: float = 1e-4
    initial_lr: float = 1e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5
    log_every: int = 10
    save_every: int = 500
    results_folder: str = "./results"
    random_seed: int = 0
    weighted_loss: bool = False
    segment_seconds: float = 2.0

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "TrainConfig":
        if not cfg:
            return cls()

        data = _section(cfg, "data")
        train = _section(cfg, "train")
        return cls(
            data_path=str(_pick(data, ("data_path", "path"), cls.data_path)),
            audio_extension=str(_pick(data, ("audio_extension",), cls.audio_extension)),
            downsampling_method=str(
                _pick(
                    data,
                    ("downsampling_method", "downsampling"),
                    cls.downsampling_method,
                )
            ),
            downsample_min=int(_pick(data, ("downsample_min",), cls.downsample_min)),
            downsample_max=int(_pick(data, ("downsample_max",), cls.downsample_max)),
            batch_size=int(_pick(train, ("batch_size", "batchsize"), cls.batch_size)),
            steps=int(_pick(train, ("steps", "n_train_steps"), cls.steps)),
            warmup_steps=int(
                _pick(train, ("warmup_steps", "n_warmup_steps"), cls.warmup_steps)
            ),
            lr=float(_pick(train, ("lr",), cls.lr)),
            initial_lr=float(_pick(train, ("initial_lr",), cls.initial_lr)),
            weight_decay=float(_pick(train, ("weight_decay", "wd"), cls.weight_decay)),
            grad_accum_steps=int(
                _pick(
                    train,
                    ("grad_accum_steps", "grad_accum_every"),
                    cls.grad_accum_steps,
                )
            ),
            max_grad_norm=float(_pick(train, ("max_grad_norm",), cls.max_grad_norm)),
            log_every=int(_pick(train, ("log_every",), cls.log_every)),
            save_every=int(
                _pick(train, ("save_every", "save_model_every"), cls.save_every)
            ),
            results_folder=str(
                _pick(train, ("results_folder", "save_dir"), cls.results_folder)
            ),
            random_seed=int(
                _pick(train, ("random_seed", "random_split_seed"), cls.random_seed)
            ),
            weighted_loss=bool(_pick(train, ("weighted_loss",), cls.weighted_loss)),
            segment_seconds=float(
                _pick(train, ("segment_seconds",), cls.segment_seconds)
            ),
        )


@dataclass(frozen=True)
class FlowHighConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "FlowHighConfig":
        cfg = {} if cfg is None else cfg
        audio = AudioConfig.from_mapping(cfg)
        return cls(
            audio=audio,
            network=NetworkConfig.from_mapping(cfg, dim_in=audio.n_mels),
            flow=FlowConfig.from_mapping(cfg),
            train=TrainConfig.from_mapping(cfg),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FlowHighConfig":
        with open(path) as f:
            return cls.from_mapping(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
