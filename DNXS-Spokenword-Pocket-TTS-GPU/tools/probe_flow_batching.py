"""Probe batched FlowLM latent steps without changing public TTS generation."""

from __future__ import annotations

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


PROBE_TEXT = "A short batched inference probe tests independent sequence items."


def _clone_batched_state(state: dict[str, dict[str, torch.Tensor]], batch_size: int):
    """Duplicate one voice-conditioned state across batch cache slots."""
    result = {}
    for module_name, module_state in state.items():
        result[module_name] = {}
        for key, value in module_state.items():
            if key == "cache":
                result[module_name][key] = value.repeat(1, batch_size, *([1] * (value.ndim - 2)))
            else:
                result[module_name][key] = value.clone()
    return result


def _measure_step(model: TTSModel, state: dict[str, Any], batch_size: int) -> float:
    """Run one batched FlowLM latent step and return synchronized duration."""
    prepared = model.flow_lm.conditioner.prepare(PROBE_TEXT)
    tokens = prepared.tokens.repeat(batch_size, 1)
    text_embeddings = model.flow_lm.conditioner(TokenizedText(tokens))
    sequence = torch.full(
        (batch_size, 1, model.flow_lm.ldim),
        float("nan"),
        device=model.device,
        dtype=model.flow_lm.dtype,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        _, eos = model.flow_lm._sample_next_latent(
            sequence=sequence,
            text_embeddings=text_embeddings,
            model_state=state,
            lsd_decode_steps=4,
            temp=0.7,
            noise_clamp=model.noise_clamp,
            eos_threshold=model.eos_threshold,
        )
    torch.cuda.synchronize()
    if eos.shape[0] != batch_size:
        raise RuntimeError(f"Expected {batch_size} EOS values, received {tuple(eos.shape)}")
    return time.perf_counter() - started


def main() -> None:
    """Load Pocket TTS and report FlowLM batch-step feasibility."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batching probe")

    model = TTSModel.load_model(device="cuda")
    base_state = model.get_state_for_audio_prompt("alba", truncate=True)
    results = []
    for batch_size in (1, 2, 4):
        state = _clone_batched_state(base_state, batch_size)
        warmup = _measure_step(model, state, batch_size)
        measured = [_measure_step(model, state, batch_size) for _ in range(3)]
        results.append(
            {
                "batch_size": batch_size,
                "warmup_seconds": warmup,
                "measured_seconds": measured,
                "mean_seconds": sum(measured) / len(measured),
                "seconds_per_item": sum(measured) / len(measured) / batch_size,
            }
        )

    print(json.dumps({"gpu_name": torch.cuda.get_device_name(0), "results": results}, indent=2))


if __name__ == "__main__":
    main()
