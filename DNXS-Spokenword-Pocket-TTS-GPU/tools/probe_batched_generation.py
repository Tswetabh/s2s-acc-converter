"""Probe fixed-horizon batched FlowLM plus Mimi generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import increment_steps, init_states


PROBE_TEXT = "A fixed horizon batching probe measures FlowLM and Mimi throughput."


def _clone_flow_state(state: dict[str, dict[str, torch.Tensor]], batch_size: int):
    """Duplicate batch-one FlowLM cache tensors across batch slots."""
    result = {}
    for module_name, module_state in state.items():
        result[module_name] = {}
        for key, value in module_state.items():
            if key == "cache":
                repeats = [1, batch_size] + [1] * (value.ndim - 2)
                result[module_name][key] = value.repeat(*repeats)
            else:
                result[module_name][key] = value.clone()
    return result


def _seed(seed: int | None) -> None:
    """Reset CUDA randomness before each comparable fixed-horizon run."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _run_batch(
    model: TTSModel,
    base_flow_state: dict[str, dict[str, torch.Tensor]],
    batch_size: int,
    steps: int,
    seed: int | None,
) -> tuple[float, torch.Tensor, list[int], list[int], int]:
    """Generate fixed latent/audio steps for one batch and return elapsed time."""
    _seed(seed)
    flow_state = _clone_flow_state(base_flow_state, batch_size)
    mimi_state = init_states(model.mimi, batch_size=batch_size, sequence_length=1000, device=model.device)
    prepared = model.flow_lm.conditioner.prepare(PROBE_TEXT)
    text_tokens = prepared.tokens.repeat(batch_size, 1)
    sequence = torch.full(
        (batch_size, 1, model.flow_lm.ldim),
        float("nan"),
        device=model.device,
        dtype=model.flow_lm.dtype,
    )
    backbone_input = sequence
    empty_tokens = torch.empty(
        (batch_size, 0), device=model.device, dtype=torch.int64
    )
    empty_conditioning = torch.empty(
        (batch_size, 0, model.flow_lm.dim),
        device=model.device,
        dtype=model.flow_lm.dtype,
    )
    audio_frames = []
    eos_steps = [-1] * batch_size

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model._run_flow_lm_and_increment_step(
            flow_state,
            text_tokens=text_tokens,
            backbone_input_latents=torch.empty(
                (batch_size, 0, model.flow_lm.ldim),
                device=model.device,
                dtype=model.flow_lm.dtype,
            ),
            audio_conditioning=empty_conditioning,
        )
        for step in range(steps):
            backbone_input, eos = model._run_flow_lm_and_increment_step(
                flow_state,
                text_tokens=empty_tokens,
                backbone_input_latents=backbone_input,
                audio_conditioning=empty_conditioning,
            )
            mimi_input = backbone_input * model.flow_lm.emb_std + model.flow_lm.emb_mean
            quantized = model.mimi.quantizer(mimi_input.transpose(-1, -2))
            audio_frame = model.mimi.decode_from_latent(quantized, mimi_state)
            audio_frames.append(audio_frame)
            increment_steps(model.mimi, mimi_state, increment=16)
            eos_values = eos.view(-1).bool().tolist()
            for index, value in enumerate(eos_values):
                if value and eos_steps[index] < 0:
                    eos_steps[index] = step
            if all(step >= 0 for step in eos_steps) and step >= max(eos_steps) + 1:
                break
    torch.cuda.synchronize()

    audio = torch.cat(audio_frames, dim=-1)
    frame_samples = audio.shape[-1] // len(audio_frames)
    audio_lengths = [
        (step + 2 if step >= 0 else len(audio_frames)) * frame_samples
        for step in eos_steps
    ]
    return time.perf_counter() - started, audio, eos_steps, audio_lengths, len(audio_frames)


def main() -> None:
    """Measure fixed-horizon batched generation at batch sizes one, two, and four."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eos-aware", action="store_true")
    parser.add_argument("--quality-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched generation probe")

    model = TTSModel.load_model(device="cuda")
    base_flow_state = model.get_state_for_audio_prompt("alba", truncate=True)
    results: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4):
        run_steps = int((len(PROBE_TEXT.split()) + 2) * 12.5) if args.eos_aware else args.steps
        warmup_times = [
            _run_batch(model, base_flow_state, batch_size, run_steps, args.seed)
            for _ in range(args.warmup)
        ]
        measured = [
            _run_batch(model, base_flow_state, batch_size, run_steps, args.seed)
            for _ in range(args.iterations)
        ]
        measured_times = [elapsed for elapsed, _, _, _, _ in measured]
        final_audio = measured[-1][1]
        quality_batch_dir = None
        if args.quality_dir is not None:
            import scipy.io.wavfile

            quality_batch_dir = args.quality_dir / f"batch_{batch_size}"
            audio_dir = quality_batch_dir / "audio_chunks"
            text_dir = quality_batch_dir / "text_chunks"
            audio_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            for index, audio_length in enumerate(measured[-1][3]):
                audio_path = audio_dir / f"chunk_{index:05d}.wav"
                audio_np = (
                    final_audio[index, :, :audio_length]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .squeeze()
                    .clip(-1.0, 1.0)
                )
                scipy.io.wavfile.write(str(audio_path), model.sample_rate, audio_np)
                (text_dir / f"chunk_{index:05d}.txt").write_text(
                    PROBE_TEXT + "\n", encoding="utf-8"
                )
        results.append(
            {
                "batch_size": batch_size,
                "steps": run_steps,
                "eos_aware": args.eos_aware,
                "warmup_seconds": [elapsed for elapsed, _, _, _, _ in warmup_times],
                "active_seconds": measured_times,
                "mean_seconds": sum(measured_times) / len(measured_times),
                "seconds_per_item": sum(measured_times) / len(measured_times) / batch_size,
                "audio_shape": list(final_audio.shape),
                "audio_finite": bool(torch.isfinite(final_audio).all().item()),
                "eos_steps": measured[-1][2],
                "audio_lengths": measured[-1][3],
                "decoded_steps": measured[-1][4],
                "quality_dir": str(quality_batch_dir) if quality_batch_dir else None,
            }
        )

    print(json.dumps({"gpu_name": torch.cuda.get_device_name(0), "results": results}, indent=2))


if __name__ == "__main__":
    main()
