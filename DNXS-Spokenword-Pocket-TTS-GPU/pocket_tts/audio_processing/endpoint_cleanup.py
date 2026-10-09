"""Combined RMS and Silero-VAD cleanup for generated audio tails."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

_VAD_MODEL: Any = None
_VAD_UTILS: Any = None
_VAD_LOAD_FAILED = False


def _mono_cpu(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio to one-dimensional CPU float samples for analysis."""
    samples = audio.detach().float().cpu()
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return samples.mean(dim=0)
    raise ValueError(f"Expected one- or two-dimensional audio, got {samples.shape}")


def _resample_for_vad(samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample mono audio to Silero VAD's required 16 kHz rate."""
    if sample_rate == 16000:
        return samples
    target_length = max(1, round(samples.numel() * 16000 / sample_rate))
    return F.interpolate(
        samples.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


def _rms_endpoint(samples: torch.Tensor, sample_rate: int, threshold: float) -> Optional[int]:
    """Find final energetic audio position using overlapping RMS windows."""
    window_size = max(1, int(0.05 * sample_rate))
    hop_size = max(1, window_size // 2)
    if samples.numel() < window_size:
        return None

    rms_values = []
    for start in range(0, samples.numel() - window_size + 1, hop_size):
        window = samples[start:start + window_size]
        rms_values.append(float(torch.sqrt(torch.mean(window.square()))))

    strong_threshold = threshold * 3.0
    for index in range(len(rms_values) - 1, -1, -1):
        if rms_values[index] <= strong_threshold:
            continue
        windows_ahead = min(10, len(rms_values) - index)
        speech_count = sum(
            value > threshold
            for value in rms_values[index:index + windows_ahead]
        )
        if speech_count >= max(1, windows_ahead * 0.3):
            return min(samples.numel(), index * hop_size + window_size)

    for index in range(len(rms_values) - 1, -1, -1):
        if rms_values[index] > threshold * 2.0:
            return min(samples.numel(), index * hop_size + window_size)
    return None


def _load_vad() -> tuple[Any, Any]:
    """Load and cache Silero VAD model and utility functions per process."""
    global _VAD_MODEL, _VAD_UTILS, _VAD_LOAD_FAILED
    if _VAD_MODEL is not None and _VAD_UTILS is not None:
        return _VAD_MODEL, _VAD_UTILS
    if _VAD_LOAD_FAILED:
        raise RuntimeError("Silero VAD model previously failed to load")

    try:
        _VAD_MODEL, _VAD_UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        _VAD_MODEL = _VAD_MODEL.cpu()
        return _VAD_MODEL, _VAD_UTILS
    except Exception:
        _VAD_LOAD_FAILED = True
        raise


def _silero_endpoint(samples: torch.Tensor, sample_rate: int) -> Optional[int]:
    """Find final speech position with Silero VAD and map it to source samples."""
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    vad_samples = _resample_for_vad(samples, sample_rate)
    speech_segments = get_speech_timestamps(
        vad_samples,
        model,
        sampling_rate=16000,
    )
    if not speech_segments:
        return None

    vad_end = speech_segments[-1]["end"]
    return min(samples.numel(), round(vad_end * sample_rate / 16000))


def cleanup_audio_endpoint(
    audio: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.004,
    buffer_ms: int = 100,
    use_silero: bool = True,
) -> torch.Tensor:
    """Trim generated tail using RMS and Silero VAD, preserving a safety buffer.

    RMS supplies a deterministic endpoint and Silero supplies speech-aware
    confirmation. When both are available, the later endpoint is retained to
    avoid cutting quiet final phonemes. If Silero cannot load, RMS cleanup
    continues and logs the fallback. Original tensor shape, dtype, and device
    are preserved.
    """
    if audio.numel() == 0:
        return audio

    samples = _mono_cpu(audio)
    rms_end = _rms_endpoint(samples, sample_rate, threshold)
    vad_end = None
    if use_silero:
        try:
            vad_end = _silero_endpoint(samples, sample_rate)
        except Exception as exc:
            logger.warning("Silero VAD unavailable; using RMS endpoint only: %s", exc)

    endpoints = [endpoint for endpoint in (rms_end, vad_end) if endpoint is not None]
    if not endpoints:
        logger.warning("No speech endpoint detected; leaving audio unchanged")
        return audio

    endpoint = max(endpoints)
    buffer_samples = max(0, int(sample_rate * buffer_ms / 1000))
    trim_samples = min(audio.shape[-1], endpoint + buffer_samples)
    logger.debug(
        "Audio endpoint cleanup: rms=%s, silero=%s, selected=%s, trim=%s/%s samples",
        rms_end,
        vad_end,
        endpoint,
        trim_samples,
        audio.shape[-1],
    )

    return audio[..., :trim_samples]
