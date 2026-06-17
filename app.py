from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gradio as gr

# Allow running from a repo checkout without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


IMPORT_ERROR: Exception | None = None
try:
    from flowhigh import FlowHighSR
    from flowhigh.postprocessing import PostProcessing
except Exception as exc:  # keep UI alive and show a useful error on load/run
    IMPORT_ERROR = exc
    FlowHighSR = None  # type: ignore[assignment]
    PostProcessing = None  # type: ignore[assignment]


CFM_METHODS = [
    "basic_cfm",
    "independent_cfm_adaptive",
    "independent_cfm_constant",
    "independent_cfm_mix",
]

MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
MODEL_LOCK = threading.Lock()

DEVICE: torch.device
ARGS: argparse.Namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowHigh Gradio WebUI")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link."
    )
    parser.add_argument(
        "--server-name", default="127.0.0.1", help="Gradio server host."
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Gradio server port."
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0.")
    parser.add_argument(
        "--ckpt-dir",
        default="",
        help="Optional local checkpoint directory containing FlowHigh model/vocoder files.",
    )
    parser.add_argument(
        "--autoload",
        action="store_true",
        help="Load the selected model when the app starts.",
    )
    args, _ = parser.parse_known_args()
    return args


def setup_device(device_str: str) -> torch.device:
    if not torch.cuda.is_available():
        # The upstream project hard-codes several .cuda() calls.
        return torch.device("cpu")

    device = torch.device(device_str)
    if device.type != "cuda":
        return torch.device("cpu")

    index = 0 if device.index is None else device.index
    torch.cuda.set_device(index)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return torch.device(f"cuda:{index}")


def require_ready() -> None:
    if IMPORT_ERROR is not None:
        raise gr.Error(
            "Could not import FlowHigh. Run this UI from the repo root or install the "
            f"package first.\n\nImport error: {IMPORT_ERROR}"
        )

    if FlowHighSR is None:
        raise gr.Error("FlowHighSR is unavailable.")

    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        raise gr.Error(
            "CUDA is required. The current FlowHigh implementation uses hard-coded "
            ".cuda() calls in the vocoder/post-processing path."
        )


def validate_local_ckpt_dir(ckpt_dir: str) -> Path:
    path = Path(ckpt_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise gr.Error(f"Local checkpoint directory does not exist: {path}")

    required = [
        "FLowHigh_basic_400k.pt",
        "bigvgan_48khz_256band.json",
        "bigvgan_48khz_256band.pt",
    ]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise gr.Error(
            "Local checkpoint directory is missing required files:\n"
            + "\n".join(f"- {name}" for name in missing)
        )

    return path


def model_cache_key(source: str, local_ckpt_dir: str) -> tuple[str, str, str]:
    if source == "Local checkpoint directory":
        local_path = str(Path(local_ckpt_dir).expanduser().resolve())
    else:
        local_path = ""
    return (source, local_path, str(DEVICE))


def load_model(
    source: str,
    local_ckpt_dir: str,
    upsampling_method: str = "scipy",
    force_reload: bool = False,
) -> Any:
    require_ready()

    key = model_cache_key(source, local_ckpt_dir)

    if not force_reload and key in MODEL_CACHE:
        model = MODEL_CACHE[key]
        model.upsampling_method = upsampling_method
        return model

    with MODEL_LOCK:
        if not force_reload and key in MODEL_CACHE:
            model = MODEL_CACHE[key]
            model.upsampling_method = upsampling_method
            return model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if source == "Hugging Face pretrained":
            model = FlowHighSR.from_pretrained(DEVICE)  # type: ignore[union-attr]
        elif source == "Local checkpoint directory":
            ckpt_path = validate_local_ckpt_dir(local_ckpt_dir)
            model = FlowHighSR.from_local(ckpt_path, DEVICE)  # type: ignore[union-attr]
        else:
            raise gr.Error(f"Unknown model source: {source}")

        # FlowHighSR.from_local/from_pretrained constructs PostProcessing(0).
        # Re-create it on the selected CUDA rank for non-zero devices.
        if PostProcessing is not None and DEVICE.type == "cuda":
            model.postproc = PostProcessing(DEVICE.index or 0)

        model.upsampling_method = upsampling_method
        model.eval()

        MODEL_CACHE.clear()
        MODEL_CACHE[key] = model
        return model


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)

    if audio.ndim == 0:
        raise gr.Error("Invalid audio input.")

    # Gradio usually returns shape [samples, channels] for stereo numpy audio.
    # Some loaders may return [channels, samples], so handle both.
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0] * 4:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise gr.Error(f"Expected mono/stereo audio, got shape {audio.shape}.")

    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-8:
        raise gr.Error("Input audio is silent or empty.")

    if peak > 1.0:
        audio = audio / peak

    return np.ascontiguousarray(audio.astype(np.float32))


def set_seed(seed: int) -> None:
    if seed < 0:
        return

    seed = int(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_memory_info() -> str:
    if not torch.cuda.is_available() or DEVICE.type != "cuda":
        return ""

    allocated = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(DEVICE) / (1024**3)
    return f"\nPeak CUDA memory: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved"


def load_button_fn(source: str, local_ckpt_dir: str, upsampling_method: str) -> str:
    start = time.perf_counter()
    model = load_model(
        source=source,
        local_ckpt_dir=local_ckpt_dir,
        upsampling_method=upsampling_method,
        force_reload=True,
    )
    elapsed = time.perf_counter() - start

    return (
        f"✅ Model loaded in {elapsed:.1f}s\n\n"
        f"Source: {source}\n"
        f"Device: {next(model.parameters()).device}\n"
        f"Upsampling: {upsampling_method}"
    )


def enhance_audio(
    audio_input: tuple[int, np.ndarray] | None,
    source: str,
    local_ckpt_dir: str,
    cfm_method: str,
    timesteps: int,
    upsampling_method: str,
    seed: int,
    max_duration_sec: float,
    normalize_output: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[tuple[int, np.ndarray], str]:
    if audio_input is None:
        raise gr.Error("Please upload or record audio first.")

    if cfm_method not in CFM_METHODS:
        raise gr.Error(f"Unsupported CFM method: {cfm_method}")

    sr, audio = audio_input
    audio = normalize_audio_array(audio)

    duration = len(audio) / float(sr)
    if max_duration_sec > 0 and duration > max_duration_sec:
        raise gr.Error(
            f"Input is {duration:.1f}s long. For this focused UI, please trim it to "
            f"{max_duration_sec:.1f}s or increase the safety limit."
        )

    set_seed(int(seed))

    progress(0.05, desc="Loading model")
    model = load_model(
        source=source,
        local_ckpt_dir=local_ckpt_dir,
        upsampling_method=upsampling_method,
        force_reload=False,
    )
    model.set_cfm_method(cfm_method)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    progress(0.15, desc="Running FlowHigh")
    start = time.perf_counter()

    with MODEL_LOCK:
        with torch.inference_mode():
            enhanced = model.generate(
                audio=audio,
                sr=int(sr),
                target_sampling_rate=48000,
                timestep=int(timesteps),
            )

    elapsed = time.perf_counter() - start
    progress(0.9, desc="Preparing output")

    enhanced_np = enhanced.squeeze().detach().float().cpu().numpy().astype(np.float32)
    enhanced_np = np.nan_to_num(enhanced_np, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize_output:
        peak = float(np.max(np.abs(enhanced_np))) if enhanced_np.size else 0.0
        if peak > 1e-8:
            enhanced_np = enhanced_np / peak * 0.99

    enhanced_np = np.clip(enhanced_np, -1.0, 1.0).astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_duration = len(enhanced_np) / 48000.0

    info = (
        "✅ Enhancement complete\n\n"
        f"Input sample rate: {sr} Hz\n"
        f"Input duration: {duration:.2f}s\n"
        f"Output sample rate: 48000 Hz\n"
        f"Output duration: {out_duration:.2f}s\n"
        f"CFM method: {cfm_method}\n"
        f"ODE timesteps: {int(timesteps)}\n"
        f"Upsampling method: {upsampling_method}\n"
        f"Seed: {'random' if int(seed) < 0 else int(seed)}\n"
        f"Device: {next(model.parameters()).device}\n"
        f"Elapsed: {elapsed:.2f}s"
        f"{gpu_memory_info()}"
    )

    progress(1.0, desc="Done")
    return (48000, enhanced_np), info


def build_demo() -> gr.Blocks:
    default_source = (
        "Local checkpoint directory" if ARGS.ckpt_dir else "Hugging Face pretrained"
    )

    with gr.Blocks(title="FlowHigh Audio Super-Resolution") as demo:
        gr.Markdown("""
            # FlowHigh Audio Super-Resolution

            Upload low-bandwidth / low-sample-rate audio and enhance it to 48 kHz.
            """)

        if IMPORT_ERROR is not None:
            gr.Markdown(f"""
                > ⚠️ FlowHigh import failed. The UI will open, but model loading will fail until dependencies are fixed.
                >
                > `{IMPORT_ERROR}`
                """)

        if not torch.cuda.is_available():
            gr.Markdown("""
                > ⚠️ CUDA was not detected. The upstream FlowHigh implementation currently requires CUDA.
                """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Input audio",
                    sources=["upload", "microphone"],
                    type="numpy",
                )

                run_btn = gr.Button("Enhance audio", variant="primary")

                with gr.Accordion("Model", open=False):
                    source = gr.Radio(
                        choices=[
                            "Hugging Face pretrained",
                            "Local checkpoint directory",
                        ],
                        value=default_source,
                        label="Model source",
                    )
                    local_ckpt_dir = gr.Textbox(
                        value=ARGS.ckpt_dir,
                        label="Local checkpoint directory",
                        placeholder="/path/to/checkpoint_dir",
                    )
                    load_btn = gr.Button("Load / Reload model")
                    load_status = gr.Textbox(
                        label="Model status",
                        value=(
                            f"Device: {DEVICE}"
                            if torch.cuda.is_available()
                            else "CUDA not available"
                        ),
                        lines=5,
                    )

                with gr.Accordion("Generation settings", open=True):
                    cfm_method = gr.Dropdown(
                        choices=CFM_METHODS,
                        value="basic_cfm",
                        label="CFM method",
                    )
                    timesteps = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=1,
                        step=1,
                        label="ODE timesteps",
                    )
                    upsampling_method = gr.Radio(
                        choices=["scipy", "librosa"],
                        value="scipy",
                        label="Pre-upsampling method",
                    )
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 = random)",
                    )
                    max_duration_sec = gr.Slider(
                        minimum=1,
                        maximum=180,
                        value=30,
                        step=1,
                        label="Max input duration safety limit, seconds",
                    )
                    normalize_output = gr.Checkbox(
                        value=True,
                        label="Normalize output peak to 0.99",
                    )

            with gr.Column(scale=1):
                audio_out = gr.Audio(
                    label="Enhanced output",
                    type="numpy",
                )
                info = gr.Textbox(label="Run info", lines=14)

        load_btn.click(
            fn=load_button_fn,
            inputs=[source, local_ckpt_dir, upsampling_method],
            outputs=[load_status],
        )

        run_btn.click(
            fn=enhance_audio,
            inputs=[
                audio_in,
                source,
                local_ckpt_dir,
                cfm_method,
                timesteps,
                upsampling_method,
                seed,
                max_duration_sec,
                normalize_output,
            ],
            outputs=[audio_out, info],
        )

        gr.Markdown("""
            Notes:
            - The pretrained model/vocoder are downloaded from `ResembleAI/FlowHigh` on first use.
            - Longer audio can require substantial GPU memory because the model operates on full mel sequences.
            - If results vary, set a non-negative seed.
            """)

    return demo


ARGS = parse_args()
DEVICE = setup_device(ARGS.device)

if ARGS.autoload:
    try:
        load_model(
            source=(
                "Local checkpoint directory"
                if ARGS.ckpt_dir
                else "Hugging Face pretrained"
            ),
            local_ckpt_dir=ARGS.ckpt_dir,
            upsampling_method="scipy",
            force_reload=True,
        )
    except Exception as exc:
        print(f"[WARN] Autoload failed: {exc}")

demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=ARGS.server_name,
        server_port=ARGS.server_port,
        share=ARGS.share,
    )
