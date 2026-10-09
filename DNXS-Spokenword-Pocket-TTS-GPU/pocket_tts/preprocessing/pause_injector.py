"""Render manual and punctuation pauses as independent TTS segments.

The pause plan is parsed before synthesis.  Each text event is generated as an
independent utterance and exact digital silence is placed only between complete
utterances; no completed waveform is cut, aligned, or energy-snapped.
"""

from __future__ import annotations

import logging
import re

from beartype.typing import Any, Callable, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

PAUSE_RE = re.compile(r"\[([\d.]+)s\]")
_PUNCT_ORDER = ["...", "--", ";", ":", ".", "!", "?", ","]
_PUNCT_RE = re.compile("|".join(re.escape(mark) for mark in _PUNCT_ORDER))
_PAUSE_OR_PUNCT_RE = re.compile(
    rf"{PAUSE_RE.pattern}|" + "|".join(re.escape(mark) for mark in _PUNCT_ORDER)
)

PauseEvents = List[Tuple[str, Any]]
SegmentRenderer = Callable[[str], torch.Tensor]


def has_inline_pause_markers(text: str) -> bool:
    """Return whether text contains one supported manual ``[Xs]`` marker."""
    return bool(PAUSE_RE.search(text or ""))


def inject_pauses_for_punctuation(text: str, pause_map: Dict[str, float]) -> str:
    """Add automatic ``[Xs]`` events after configured punctuation.

    A manually authored marker immediately after punctuation always wins, so
    enabling punctuation pauses cannot duplicate a requested manual pause.
    """
    def annotate(match: re.Match[str]) -> str:
        """Return punctuation with one automatic marker when configured."""
        punctuation = match.group(0)
        if PAUSE_RE.fullmatch(punctuation):
            # Consume existing markers as one token so their decimal points are
            # never mistaken for punctuation that needs another marker.
            return punctuation
        seconds = float(pause_map.get(punctuation, 0.0))
        following = text[match.end() :]
        if seconds <= 0 or PAUSE_RE.match(following.lstrip()):
            return punctuation
        return f"{punctuation}[{seconds:.2f}s]"

    return _PAUSE_OR_PUNCT_RE.sub(annotate, text or "")


def parse_text_with_pauses(raw: str) -> tuple[PauseEvents, List[Dict[str, Any]]]:
    """Parse ``[Xs]`` text into ordered text and pause events.

    Returns:
        Ordered events and pause records retaining source positions.
    """
    events: PauseEvents = []
    pauses: List[Dict[str, Any]] = []
    cursor = 0
    for index, match in enumerate(PAUSE_RE.finditer(raw or "")):
        text = raw[cursor : match.start()]
        if text:
            events.append(("text", text))
        seconds = float(match.group(1))
        events.append(("pause", seconds))
        pauses.append({
            "index": index,
            "seconds": seconds,
            "start_char": match.start(),
            "end_char": match.end(),
        })
        cursor = match.end()
    tail = (raw or "")[cursor:]
    if tail:
        events.append(("text", tail))
    return events, pauses


def render_text_with_native_pauses(raw: str) -> str:
    """Return text segments joined for human-readable spoken-text sidecars.

    This is metadata only.  It deliberately does not add commas because TTS
    receives each segment independently when an inline marker is present.
    """
    events, _ = parse_text_with_pauses(raw)
    return " ".join(str(value).strip() for kind, value in events if kind == "text" and str(value).strip())


def is_speech_text_segment(text: str) -> bool:
    """Return whether a pause text event contains spoken alphanumeric content.

    Quote-only and punctuation-only fragments remain represented by the source
    pause plan, but sending them as independent TTS prompts creates context-free
    hallucinations at chunk boundaries.
    """
    return any(character.isalnum() for character in str(text or ""))


def _empty_audio(sample_count: int, device: Any = None) -> torch.Tensor:
    """Create a float32 fallback waveform for a marker-only pause plan."""
    try:
        return torch.zeros(sample_count, dtype=torch.float32, device=device)
    except (RuntimeError, TypeError):
        return torch.zeros(sample_count, dtype=torch.float32)


def render_pause_events(
    events: PauseEvents,
    render_segment: SegmentRenderer,
    sample_rate: int,
    *,
    fallback_device: Any = None,
) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Generate text events and concatenate them with exact zero-sample pauses.

    Args:
        events: Ordered ``("text", value)`` and ``("pause", seconds)`` plan.
        render_segment: Callback that returns fully postprocessed speech for one
            nonempty text segment.
        sample_rate: Sample rate used to convert requested seconds to samples.
        fallback_device: Device for a marker-only plan with no speech tensor.

    Returns:
        Final waveform and serializable records for text and pause events.
    """
    rendered: List[Tuple[str, Any]] = []
    records: List[Dict[str, Any]] = []
    reference_audio: torch.Tensor | None = None

    for kind, value in events:
        if kind == "text":
            text = str(value).strip()
            if not text or not is_speech_text_segment(text):
                continue
            audio = render_segment(text)
            if not isinstance(audio, torch.Tensor):
                raise TypeError("Pause segment renderer must return a torch.Tensor")
            rendered.append(("audio", audio))
            reference_audio = reference_audio if reference_audio is not None else audio
            records.append({"kind": "text", "text": text, "samples": int(audio.shape[-1])})
            continue
        seconds = max(0.0, float(value))
        samples = max(0, int(round(seconds * sample_rate)))
        rendered.append(("pause", (seconds, samples)))
        records.append({"kind": "pause", "seconds": seconds, "samples": samples})

    if reference_audio is None:
        total_samples = sum(item[1][1] for item in rendered if item[0] == "pause")
        return _empty_audio(total_samples, fallback_device), records

    pieces: List[torch.Tensor] = []
    for kind, value in rendered:
        if kind == "audio":
            pieces.append(value)
            continue
        _seconds, samples = value
        pieces.append(reference_audio.new_zeros((*reference_audio.shape[:-1], samples)))
    if not pieces:
        return reference_audio.new_zeros((*reference_audio.shape[:-1], 0)), records
    return torch.cat(pieces, dim=-1), records


def generate_audio_with_pauses(
    tts_model: Any,
    voice_state: Any,
    raw_text: str,
    *,
    generate_segment: SegmentRenderer | None = None,
) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Generate a pause plan as separate speech segments plus digital silence.

    ``generate_segment`` lets each caller apply its own TTS parameters, speed
    adjustment, and endpoint cleanup before segment assembly.
    """
    from pocket_tts.preprocessing.text_normalizer import flatten_newlines_for_json

    plan_text = flatten_newlines_for_json(raw_text or "")
    events, _ = parse_text_with_pauses(plan_text)
    if generate_segment is None:
        def generate_segment(text: str) -> torch.Tensor:
            """Generate one speech event with the model's default settings."""
            return tts_model.generate_audio(voice_state, text)

    logger.debug("Pause plan events: %r", events[:10])
    return render_pause_events(
        events,
        generate_segment,
        int(getattr(tts_model, "sample_rate", 24000)),
        fallback_device=getattr(tts_model, "device", None),
    )
