"""
Pause injection for punctuation-based silence.

Converts punctuation marks to pause markers [Xs], then generates audio
with digital silence at those positions.
"""

import re
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

PAUSE_RE = re.compile(r"\[([\d.]+)s\]")

# Order matters: longer patterns first to avoid partial replacements
_PUNCT_ORDER = ["...", "--", ";", ":", ".", "!", "?", ","]
_PUNCT_RE = re.compile("|".join(re.escape(punctuation) for punctuation in _PUNCT_ORDER))


def has_inline_pause_markers(text: str) -> bool:
    """Check if text contains [Xs] pause markers."""
    return bool(PAUSE_RE.search(text))


def inject_pauses_for_punctuation(text: str, pause_map: Dict[str, float]) -> str:
    """Annotate punctuation with pause metadata while preserving spoken punctuation.

    The markers are metadata only.  TTS receives the complete punctuated text
    in one request so punctuation pauses cannot fragment a chunk into tiny,
    context-free prompts.
    """
    def annotate(match: re.Match[str]) -> str:
        """Append one marker unless text already supplies a manual pause."""
        punctuation = match.group(0)
        seconds = float(pause_map.get(punctuation, 0.0))
        following = text[match.end() :]
        if seconds <= 0 or PAUSE_RE.match(following):
            return punctuation
        return f"{punctuation}[{seconds:.2f}s]"

    return _PUNCT_RE.sub(annotate, text)


def parse_text_with_pauses(raw: str):
    """Parse text containing [Xs] pause markers into events.

    Returns:
        events: List of ("text", str) or ("pause", float) tuples
        pauses: List of pause metadata dicts with index, seconds, positions
    """
    events: List[Tuple[str, Any]] = []
    pauses: List[Dict[str, Any]] = []

    cursor = 0
    pause_index = 0

    for m in PAUSE_RE.finditer(raw):
        # Add preceding text chunk
        chunk = raw[cursor : m.start()]
        if chunk:
            events.append(("text", chunk))

        # Add pause
        seconds = float(m.group(1))
        events.append(("pause", seconds))
        pauses.append(
            {
                "index": pause_index,
                "seconds": seconds,
                "start_char": m.start(),
                "end_char": m.end(),
            }
        )
        pause_index += 1

        cursor = m.end()

    # Add trailing text
    tail = raw[cursor:]
    if tail:
        events.append(("text", tail))

    return events, pauses


def render_text_with_native_pauses(raw: str) -> str:
    """Remove inline markers while retaining a single natural TTS prompt.

    Markers following punctuation are removed because the model already sees
    its native punctuation cue.  A manually supplied marker embedded between
    words becomes a comma so words are never accidentally joined together.
    """
    def replace_marker(match: re.Match[str]) -> str:
        """Choose a safe spoken separator for one marker occurrence."""
        previous = raw[match.start() - 1] if match.start() else ""
        following = raw[match.end()] if match.end() < len(raw) else ""
        if previous.isspace() or following.isspace() or previous in ".!?;:,":
            return ""
        # Explicit markers can appear directly between words; retain a pause cue.
        return ", "

    return PAUSE_RE.sub(replace_marker, raw)


_TERMINAL_PUNCTUATION = ".!?\u2026"


def ensure_terminal_punctuation(text: str) -> str:
    """Guarantee a stop cue so the model does not loop on punctuation-less text.

    Short prompts without terminal punctuation make the autoregressive model
    repeat the utterance (e.g. "Chapter One Chapter One" ghosting).  Appends a
    period only when the spoken text has no final punctuation.
    """
    stripped = text.strip()
    if not stripped:
        return text
    if stripped[-1] in _TERMINAL_PUNCTUATION:
        return text
    return stripped + "."


def _text_anchor_ratio(raw_text: str, marker_start: int) -> float:
    """Estimate marker timing from its source-text position after marker removal."""
    spoken_text = PAUSE_RE.sub("", raw_text)
    if not spoken_text:
        return 1.0

    preceding_text = PAUSE_RE.sub("", raw_text[:marker_start])
    return min(1.0, max(0.0, len(preceding_text) / len(spoken_text)))


def _snap_to_low_energy_boundary(
    audio: torch.Tensor,
    estimated_position: int,
    sample_rate: int,
) -> int:
    """Snap forward from a text anchor to the next quiet audio boundary."""
    samples = audio if audio.ndim == 1 else audio.mean(dim=0)
    sample_count = int(samples.shape[-1])
    if sample_count == 0:
        return 0

    radius = max(1, int(sample_rate * 0.45))
    frame_size = max(1, int(sample_rate * 0.02))
    hop_size = max(1, int(sample_rate * 0.005))
    start = min(sample_count, max(0, estimated_position))
    end = min(sample_count, estimated_position + radius)
    segment = samples[start:end]
    if segment.numel() < frame_size:
        return min(sample_count, max(0, estimated_position))

    energy = F.avg_pool1d(
        segment.square().reshape(1, 1, -1),
        kernel_size=frame_size,
        stride=hop_size,
    ).reshape(-1)
    minimum_energy = energy.min()
    quietest_frames = torch.nonzero(
        torch.isclose(energy, minimum_energy),
        as_tuple=False,
    ).reshape(-1)
    quietest_frame = int(quietest_frames[0].item())
    return min(sample_count, start + quietest_frame * hop_size + frame_size // 2)


def insert_inline_pauses_into_audio(
    audio: torch.Tensor,
    raw_text: str,
    sample_rate: int,
) -> torch.Tensor:
    """Insert requested silence into audio without fragmenting its TTS prompt.

    Markers are mapped from source-text position, then moved only forward to
    the next low-energy speech boundary. This preserves one full-context model
    request while preventing a pause from moving before its punctuation.
    """
    _, pauses = parse_text_with_pauses(raw_text)
    if not pauses:
        return audio
    if audio.numel() == 0:
        # Marker-only chunk (no spoken text): synthesize the requested silence
        # directly.  Without this, TTS produces a 0-sample chunk, the WAV write
        # emits a header-only file, and the ASR pipeline crashes on it.
        total_seconds = sum(
            max(0.0, float(pause.get("seconds", 0.0))) for pause in pauses
        )
        total_samples = max(1, int(round(total_seconds * sample_rate)))
        return audio.new_zeros((*audio.shape[:-1], total_samples))

    original_samples = int(audio.shape[-1])
    inserted_samples = 0
    pieces: List[torch.Tensor] = []
    source_cursor = 0

    for pause in pauses:
        seconds = float(pause["seconds"])
        pause_samples = max(0, int(round(seconds * sample_rate)))
        if pause_samples == 0:
            continue

        if raw_text[pause["end_char"] :].strip():
            anchor_ratio = _text_anchor_ratio(raw_text, pause["start_char"])
            estimated_position = int(round(original_samples * anchor_ratio))
            if estimated_position == 0:
                # Marker at the very start of the chunk: the pause belongs
                # before all generated speech. Snapping forward here would
                # hunt for the quietest frame inside the first words and
                # split them ("Chapter [5s] Three").
                source_position = 0
            else:
                source_position = _snap_to_low_energy_boundary(
                    audio,
                    estimated_position,
                    sample_rate,
                )
        else:
            # Sentence-ending pauses belong after all generated speech/tail.
            source_position = original_samples
        source_position = min(original_samples, max(source_cursor, source_position))
        pieces.append(audio[..., source_cursor:source_position])
        pieces.append(audio.new_zeros((*audio.shape[:-1], pause_samples)))
        source_cursor = source_position
        inserted_samples += pause_samples

    if inserted_samples == 0:
        return audio
    pieces.append(audio[..., source_cursor:])
    return torch.cat(pieces, dim=-1)


def generate_audio_with_pauses(
    tts_model,
    voice_state,
    raw_text: str,
    insert_pauses: bool = True,
):
    """Generate one full-context audio prompt, optionally inserting pauses.

    Digital splicing used to generate each text segment independently.  That
    removed surrounding linguistic context and caused a large ASR failure
    increase. Callers with endpoint cleanup can defer silence insertion until
    after trimming by setting ``insert_pauses`` to ``False``.
    """
    events, pauses = parse_text_with_pauses(raw_text)
    logger.debug(f"Injected text: {repr(raw_text[:100])}")
    logger.debug(f"Parsed events: {events[:10]}")  # First 10 events
    logger.debug(f"Parsed pauses: {pauses}")
    spoken_text = render_text_with_native_pauses(raw_text).strip()
    spoken_text = ensure_terminal_punctuation(spoken_text)
    if not spoken_text:
        return torch.zeros(0, dtype=torch.float32), pauses
    logger.debug(f"Generating one TTS prompt with native pauses: {repr(spoken_text[:50])}")
    audio = tts_model.generate_audio(voice_state, spoken_text)
    if insert_pauses:
        audio = insert_inline_pauses_into_audio(
            audio,
            raw_text,
            getattr(tts_model, "sample_rate", 24000),
        )
    return audio, pauses
