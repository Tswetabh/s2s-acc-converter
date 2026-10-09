"""Benchmark isolated Pocket TTS precision and compilation variants."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.models.tts_model import TTSModel


VARIANT_TEXT = (
    "This controlled workload measures repeated autoregressive generation and "
    "audio decoding for one stable voice prompt."
)


def _synchronize() -> None:
    """Wait for queued CUDA work before recording elapsed time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_generation(seed: int | None) -> None:
    """Reset CPU and CUDA RNGs before a comparable generation run."""
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_context(precision: str):
    """Return the requested CUDA autocast context or a no-op context."""
    if precision == "none":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _compile_target(model: TTSModel, target: str, mode: str) -> None:
    """Compile one isolated model component without changing production code."""
    if target == "flow_net":
        model.flow_lm.flow_net = torch.compile(model.flow_lm.flow_net, mode=mode)
    elif target == "flow_step":
        model.flow_lm._sample_next_latent = torch.compile(
            model.flow_lm._sample_next_latent,
            mode=mode,
        )
    elif target == "mimi_decode":
        model.mimi.decode_from_latent = torch.compile(
            model.mimi.decode_from_latent,
            mode=mode,
        )
    elif target != "none":
        raise ValueError(f"Unknown compile target: {target}")


def _generate(
    model: TTSModel,
    voice_state: dict[str, Any],
    precision: str,
    seed: int | None,
) -> tuple[float, torch.Tensor]:
    """Generate one audio sample and return synchronized time plus audio."""
    _seed_generation(seed)
    _synchronize()
    started = time.perf_counter()
    with _autocast_context(precision):
        audio = model.generate_audio(voice_state, VARIANT_TEXT, frames_after_eos=2)
    _synchronize()
    return time.perf_counter() - started, audio


def run_variant(
    precision: str,
    compile_target: str,
    compile_mode: str,
    matmul_precision: str,
    warmup_count: int,
    iteration_count: int,
    output_path: Path | None,
    quality_dir: Path | None,
    seed: int | None,
) -> dict[str, Any]:
    """Load one model variant and measure warmup and steady-state generation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference variant benchmarks")

    torch.set_float32_matmul_precision(matmul_precision)
    model = TTSModel.load_model(device="cuda")
    _compile_target(model, compile_target, compile_mode)
    voice_state = model.get_state_for_audio_prompt("alba", truncate=True)

    warmup_times = []
    for _ in range(warmup_count):
        elapsed, _ = _generate(model, voice_state, precision, seed)
        warmup_times.append(elapsed)

    active_times = []
    final_audio = None
    for _ in range(iteration_count):
        elapsed, final_audio = _generate(model, voice_state, precision, seed)
        active_times.append(elapsed)

    if final_audio is None:
        raise RuntimeError("Variant produced no audio")
    audio_float = final_audio.detach().float()
    audio_finite = bool(torch.isfinite(audio_float).all().item())
    audio_peak = float(audio_float.abs().max().item()) if audio_float.numel() else 0.0
    audio_duration = audio_float.numel() / model.sample_rate

    quality_audio_path = None
    if quality_dir is not None:
        quality_audio_path = quality_dir / "audio_chunks" / "chunk_00000.wav"
        quality_text_path = quality_dir / "text_chunks" / "chunk_00000.txt"
        quality_text_path.parent.mkdir(parents=True, exist_ok=True)
        quality_text_path.write_text(VARIANT_TEXT + "\n", encoding="utf-8")
        quality_audio_path.parent.mkdir(parents=True, exist_ok=True)

    audio_targets = [path for path in (output_path, quality_audio_path) if path is not None]
    for audio_target in audio_targets:
        import scipy.io.wavfile

        audio_target.parent.mkdir(parents=True, exist_ok=True)
        audio_np = audio_float.cpu().numpy().squeeze().clip(-1.0, 1.0)
        scipy.io.wavfile.write(str(audio_target), model.sample_rate, audio_np)

    return {
        "precision": precision,
        "compile_target": compile_target,
        "compile_mode": compile_mode,
        "matmul_precision": matmul_precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "warmup_seconds": warmup_times,
        "active_seconds": active_times,
        "active_mean_seconds": sum(active_times) / len(active_times),
        "audio_duration_seconds": audio_duration,
        "audio_finite": audio_finite,
        "audio_peak": audio_peak,
        "model_timing": model.last_generation_timing,
        "output_path": str(output_path) if output_path else None,
        "quality_dir": str(quality_dir) if quality_dir else None,
        "seed": seed,
    }


def main() -> None:
    """Run one requested variant and emit machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument(
        "--compile-target",
        choices=["none", "flow_net", "flow_step", "mimi_decode"],
        default="none",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="highest",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    try:
        result = run_variant(
            args.precision,
            args.compile_target,
            args.compile_mode,
            args.matmul_precision,
            args.warmup,
            args.iterations,
            args.output,
            args.quality_dir,
            args.seed,
        )
    except Exception as exc:
        result = {
            "precision": args.precision,
            "compile_target": args.compile_target,
            "compile_mode": args.compile_mode,
            "matmul_precision": args.matmul_precision,
            "seed": args.seed,
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        result["success"] = True

    print(json.dumps(result, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
