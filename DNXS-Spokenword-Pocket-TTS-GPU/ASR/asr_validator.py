#!/usr/bin/env python3
"""
ASR Validation Tool - Standalone Tkinter GUI
Validates TTS-generated audio chunks using ASR and compares to reference text.
Generates validation.log and fail.log reports.

SELF-CONTAINED VERSION - No external dependencies on modules/
Uses faster-whisper for ASR transcription
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import re
import time
import argparse

# ASR/Similarity related imports
import rapidfuzz.fuzz as fuzz
import torch
import librosa
import numpy as np

try:
    from .spoken_compare import (
        build_book_term_evidence,
        compare_spoken,
        normalize as normalize_spoken,
    )
except ImportError:  # Supports direct execution from the ASR directory.
    from spoken_compare import build_book_term_evidence, compare_spoken, normalize as normalize_spoken

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - librosa path still works
    sf = None

# Set up basic logging for the standalone app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_ASR_MODEL = "base"
ASR_SAFETY_BUFFER_MB = 500
PASS_TOLERANCE_SCORE = 0.95  # Allow high-score chunks to pass despite minor hallucination warnings

# Configuration file for persisting user settings
CONFIG_FILE = Path(__file__).parent / "asr_config.json"

# ASR Model Memory Requirements (MB)
ASR_MODEL_VRAM_MB = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550,
    "large-v2": 1550,
    "large-v3": 1550,
    # large-v3-turbo is much lighter/faster than full large-v3 (~medium-class VRAM).
    "large-v3-turbo": 900,
    "turbo": 900,
}

def _format_device(device: Optional[str]) -> str:
    """Return a stable uppercase display name for a requested ASR device."""
    if isinstance(device, str):
        return device.upper()
    return str(device)

# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Delegate all production normalization to the shared spoken comparator."""
    return normalize_spoken(text)


def token_dp_align(
    ref_tokens: List[str], hyp_tokens: List[str]
) -> Dict[str, Any]:
    """Token-level DP alignment returning structured edit script and metrics.

    Returns dict with:
      - operations: list of {op, ref, hyp, ref_idx?, hyp_idx?}
      - matched, extra, missing, sub: counts
      - extra_list, missing_list, sub_list: token lists
      - coverage: float (matched / max(len(ref),1))
      - phonetic_score: float (phonetic matches among subs)
    """
    # placeholder - implementation later
    return {
        "operations": [],
        "matched": 0,
        "extra": 0,
        "missing": 0,
        "sub": 0,
        "extra_list": [],
        "missing_list": [],
        "sub_list": [],
        "coverage": 0.0,
        "phonetic_score": 0.0,
    }


def phonetic_equivalent(a: str, b: str) -> bool:
    """Double Metaphone primary+secondary with constrained Soundex fallback.

    Char similarity used only as tie-breaker. Explicitly rejects reed/red,
    lyve/liv unless ASR token itself is ambiguous spelling.
    """
    try:
        from metaphone import doublemetaphone
    except ImportError:
        return a == b
    pa1, pa2 = doublemetaphone(a)
    pb1, pb2 = doublemetaphone(b)
    if pa1 and pb1 and (pa1 == pb1 or pa1 == pb2 or pa2 == pb1 or pa2 == pb2):
        return True
    if not pa1 and not pb1:
        return _soundex(a) == _soundex(b)
    return False


# ============================================================================
# SIMILARITY
# ============================================================================

def similarity(a: str, b: str) -> float:
    """
    Calculate similarity ratio between two normalized strings.
    """
    return fuzz.ratio(a, b) / 100.0


def _soundex(text: str) -> str:
    """Create a compact phonetic key for fuzzy word comparisons."""
    letters = re.sub(r"[^a-z]", "", text.lower())
    if not letters:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    first = letters[0].upper()
    encoded = []
    previous = codes.get(letters[0], "")
    for letter in letters[1:]:
        code = codes.get(letter, "")
        if code and code != previous:
            encoded.append(code)
        previous = code
    return (first + "".join(encoded) + "000")[:4]


def _is_minor_mismatch(reference: str, hypothesis: str) -> bool:
    """Identify close substitutions without forgiving missing or extra speech."""
    import difflib

    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    matcher = difflib.SequenceMatcher(None, reference_tokens, hypothesis_tokens)
    substitutions = 0
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag in {"insert", "delete"}:
            return False
        if tag != "replace":
            continue
        substitutions += 1
        reference_phrase = "".join(reference_tokens[ref_start:ref_end])
        hypothesis_phrase = "".join(hypothesis_tokens[hyp_start:hyp_end])
        close_spelling = fuzz.ratio(reference_phrase, hypothesis_phrase) >= 60
        close_sound = _soundex(reference_phrase) == _soundex(hypothesis_phrase)
        if not close_spelling and not close_sound:
            return False
    return 0 < substitutions <= 2

# ============================================================================
# DIFF EXPLANATION
# ============================================================================

def explain_diff(ref: str, hyp: str) -> str:
    """
    Generate human-readable, sequence-aware explanation for similarity failure.
    Uses token-level alignment to show insertions, deletions, and substitutions.
    """
    import difflib

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    # Use SequenceMatcher for token-level alignment
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)

    insertions = []
    deletions = []
    substitutions = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            deletions.extend(ref_tokens[i1:i2])
        elif tag == 'insert':
            insertions.extend(hyp_tokens[j1:j2])
        elif tag == 'replace':
            # Treat as substitution
            ref_segment = ' '.join(ref_tokens[i1:i2])
            hyp_segment = ' '.join(hyp_tokens[j1:j2])
            substitutions.append(f"'{ref_segment}' → '{hyp_segment}'")

    # Build explanation
    parts = []
    if deletions:
        parts.append(f"missing: {', '.join(deletions)}")
    if insertions:
        parts.append(f"extra: {', '.join(insertions)}")
    if substitutions:
        parts.append(f"substituted: {'; '.join(substitutions)}")

    if parts:
        return '; '.join(parts)
    else:
        return "minor word order or spacing differences"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Format elapsed time in seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds (can be float)

    Returns:
        Formatted string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_chunks(tts_dir: Path) -> List[str]:
    """
    Find all matching audio/text chunk pairs.
    Returns sorted list of chunk identifiers (e.g., "chunk_00001")
    """
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"

    if not audio_dir.exists() or not text_dir.exists():
        return []

    audio_chunks = set()
    text_chunks = set()

    for f in audio_dir.glob("chunk_*.wav"):
        num = f.stem  # "chunk_00001"
        audio_chunks.add(num)

    for f in text_dir.glob("chunk_*.txt"):
        num = f.stem
        text_chunks.add(num)

    # Return intersection (chunks that have both)
    return sorted(audio_chunks & text_chunks)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_last_folder() -> Path:
    """
    Load the last used TTS folder from config file.
    Returns the program directory if no config exists or path is invalid.
    """
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                last_folder = config.get('last_folder')
                if last_folder and Path(last_folder).exists():
                    return Path(last_folder)
    except (json.JSONDecodeError, IOError, KeyError):
        pass  # Fall back to default

    # Default to program directory
    return Path(__file__).parent

def save_last_folder(folder_path: Path) -> None:
    """
    Save the last used TTS folder to config file.
    """
    try:
        config = {'last_folder': str(folder_path)}
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError:
        # Silently fail if we can't write config
        pass

def _load_asr_model() -> str:
    """Load persisted ASR model or default to 'base'."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model = config.get('asr_model')
                if model in ("base", "small", "medium"):
                    return model
    except (json.JSONDecodeError, IOError, KeyError):
        pass
    return DEFAULT_ASR_MODEL

def _save_asr_model(model_name: str) -> None:
    """Persist ASR model choice into config file."""
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        config['asr_model'] = model_name
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError:
        pass

# ============================================================================
# ASR MODEL LOADING
# ============================================================================

def get_real_time_vram_status():
    """Get current GPU memory usage in real-time"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Use first GPU
                total_vram = torch.cuda.get_device_properties(0).total_memory
                allocated_vram = torch.cuda.memory_allocated(0)
                reserved_vram = torch.cuda.memory_reserved(0)
                available_vram = total_vram - allocated_vram

                return {
                    'total_mb': total_vram // 1024 // 1024,
                    'allocated_mb': allocated_vram // 1024 // 1024,
                    'reserved_mb': reserved_vram // 1024 // 1024,
                    'available_mb': available_vram // 1024 // 1024,
                    'has_gpu': True
                }
    except Exception as e:
        logging.warning(f"Failed to get real-time VRAM status: {e}")

    return {
        'total_mb': 0,
        'allocated_mb': 0,
        'reserved_mb': 0,
        'available_mb': 0,
        'has_gpu': False
    }

def calculate_available_vram_for_asr(safety_buffer_mb=500):
    """Calculate VRAM available for ASR with safety buffer"""
    vram_status = get_real_time_vram_status()

    if not vram_status['has_gpu']:
        return 0

    # Available VRAM minus safety buffer for stability
    available_with_buffer = max(0, vram_status['available_mb'] - safety_buffer_mb)

    return available_with_buffer

def can_model_fit_gpu(model_name, available_vram_mb):
    """Check if a specific ASR model can fit in available VRAM"""
    required_vram = ASR_MODEL_VRAM_MB.get(model_name, 0)
    return available_vram_mb >= required_vram

def load_asr_model_adaptive(
    model_name: str = DEFAULT_ASR_MODEL,
    force_device: Optional[str] = None,
    engine: str = "faster_whisper",
    n_threads: int = 2,
    allow_cpu_fallback: bool = True,
):
    """
    Adaptive ASR model loading with real-time VRAM checking and intelligent fallback.

    Supports faster-whisper (default, including distil-* models) and whisper.cpp
    via pywhispercpp (ASR/whisper_cpp_backend.py).

    Args:
        model_name: Model size (base, small, medium, distil-small.en, …).
        force_device: Optional 'cpu' or 'cuda' override. When set, skips VRAM
            adaptive selection.
        engine: 'faster_whisper' or 'whisper_cpp'.
        n_threads: CPU threads per whisper.cpp worker (unused for faster-whisper).
        allow_cpu_fallback: When false, a forced CUDA load reports failure
            instead of silently loading a CPU model.

    Returns:
        tuple: (asr_model, actual_device_used) or (None, None) if all loading fails
    """
    eng = (engine or "faster_whisper").strip().lower().replace("-", "_")
    if eng in ("whisper_cpp", "whispercpp", "cpp", "pywhispercpp"):
        _asr_dir = str(Path(__file__).resolve().parent)
        if _asr_dir not in sys.path:
            sys.path.insert(0, _asr_dir)
        from whisper_cpp_backend import load_whisper_cpp_model

        return load_whisper_cpp_model(
            model_name=model_name,
            force_device=force_device,
            n_threads=n_threads,
        )

    from faster_whisper import WhisperModel

    try:
        from model_download_progress import (
            ensure_faster_whisper_model,
            format_load_banner,
        )

        print(
            format_load_banner(
                "faster_whisper",
                model_name,
                (force_device or "auto"),
                1,
            ),
            flush=True,
        )
        ensure_faster_whisper_model(model_name, progress_cb=None)
    except Exception as pre_exc:
        print(f"⚠️ ASR pre-download note: {pre_exc}", flush=True)

    print(f"🔍 Loading faster-whisper model: {model_name}", flush=True)

    forced = (force_device or "").strip().lower()
    if forced in ("cpu", "cuda", "gpu"):
        if forced == "gpu":
            forced = "cuda"
        if forced == "cuda":
            device = "cuda"
            compute_type = "float16"
            device_display = "GPU"
            print(f"📌 Forced device: GPU for {model_name} (float16)")
        else:
            device = "cpu"
            compute_type = "int8"
            device_display = "CPU"
            print(f"📌 Forced device: CPU for {model_name} (int8) — no GPU fight with TTS")
    else:
        # Get current VRAM status
        vram_status = get_real_time_vram_status()
        available_vram = calculate_available_vram_for_asr()

        print("🖥️ Real-time VRAM status:")
        print(f"   Total: {vram_status['total_mb']:,}MB")
        print(f"   Allocated: {vram_status['allocated_mb']:,}MB")
        print(f"   Available for ASR: {available_vram:,}MB (with 500MB safety buffer)")
        required_vram = ASR_MODEL_VRAM_MB.get(model_name, 0)
        print(f"   Estimated {model_name} requirement: {required_vram:,}MB")
        if vram_status["has_gpu"]:
            print(
                f"   GPU: {torch.cuda.get_device_name(0)} "
                f"capability={torch.cuda.get_device_capability(0)}"
            )

        # Choose device based on real-time VRAM availability
        if vram_status["has_gpu"] and can_model_fit_gpu(model_name, available_vram):
            device = "cuda"
            compute_type = "float16"
            device_display = "GPU"
            print(f"✅ Using GPU for {model_name} (float16)")
        else:
            device = "cpu"
            compute_type = "int8"
            device_display = "CPU"
            print(f"🔄 Using CPU for {model_name} (int8)")

    try:
        asr_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print(f"✅ Successfully loaded {model_name} on {device_display}")
        return asr_model, device_display.lower()
    except Exception as e:
        print(f"❌ Critical failure: Could not load {model_name} on {device}: {e}")

        # The legacy path permits CPU fallback. New isolated GPU stages use a
        # hard failure so a fallback cannot hide an exhausted CUDA boundary.
        if device == "cuda" and not allow_cpu_fallback:
            print(f"🛑 GPU-only load failed for {model_name}; CPU fallback disabled")
            return None, "cuda_load_failed"

        # Ultimate fallback to CPU if GPU failed
        if device == "cuda":
            try:
                print(f"🆘 Ultimate fallback: {model_name} on CPU")
                asr_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                print(f"✅ Successfully loaded {model_name} on CPU")
                return asr_model, "cpu"
            except Exception:
                print(f"❌ Total failure: Could not load {model_name} on any device")
                return None, None

    return None, None

def cleanup_asr_model(asr_model):
    """Clean up ASR model to free memory"""
    if asr_model is not None:
        try:
            del asr_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 ASR model cleaned up")
        except Exception as e:
            logging.warning(f"Failed to cleanup ASR model: {e}")

# ============================================================================
# HALLUCINATION & TRUNCATION DETECTION
# ============================================================================

def detect_hallucination(ref_text: str, hyp_text: str) -> dict:
    """
    Detect repetitive patterns that suggest hallucination (context-aware).
    Compares hypothesis repetitions against reference to avoid false positives.

    Args:
        ref_text: Reference text (original/expected)
        hyp_text: Hypothesis text (ASR transcription)

    Returns:
        {"is_hallucination": bool, "pattern": str, "count": int, "type": str, "severity": str}
    """
    def normalize_for_comparison(text):
        """Canonicalize sound-spelling variants before counting repeated runs."""
        # Remove common punctuation but keep letters/numbers
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join({'ha': 'ha', 'hah': 'ha', 'haw': 'ha'}.get(token, token)
                        for token in normalized.split())

    def count_max_adjacent_repeats(text):
        """Count maximum adjacent repetitions of words and phrases in text"""
        # Normalize text for fair comparison
        normalized = normalize_for_comparison(text)
        tokens = normalized.split()

        if len(tokens) < 2:
            return {}, {}

        # Single word repetitions
        word_repeats = {}
        current_word = tokens[0]
        current_count = 1

        for token in tokens[1:]:
            if token == current_word:
                current_count += 1
            else:
                if current_count >= 2:
                    word_repeats[current_word] = max(
                        word_repeats.get(current_word, 0),
                        current_count
                    )
                current_word = token
                current_count = 1

        # Check last word
        if current_count >= 2:
            word_repeats[current_word] = max(
                word_repeats.get(current_word, 0),
                current_count
            )

        # Phrase repetitions (2-word sequences)
        phrase_repeats = {}
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i+1]}"
            count = 1
            j = i + 1
            while j < len(tokens) - 1:
                next_phrase = f"{tokens[j]} {tokens[j+1]}"
                if next_phrase == phrase:
                    count += 1
                    j += 1
                else:
                    break
            if count >= 2:
                phrase_repeats[phrase] = max(phrase_repeats.get(phrase, 0), count)

        return word_repeats, phrase_repeats

    # Get repetition counts for both texts
    ref_word_repeats, ref_phrase_repeats = count_max_adjacent_repeats(ref_text)
    hyp_word_repeats, hyp_phrase_repeats = count_max_adjacent_repeats(hyp_text)

    # Check for hallucinated word repetitions
    for word, hyp_count in hyp_word_repeats.items():
        ref_count = ref_word_repeats.get(word, 0)

        # Only flag if hypothesis has MORE repetitions than reference
        if hyp_count > ref_count:
            excess = hyp_count - ref_count

            # Determine severity - classify short common words as tolerable
            short_common_words = {'the', 'a', 'to', 'in', 'as', 'of', 'on', 'at', 'for', 'by', 'with', 'from', 'is'}

            if hyp_count >= 4 or excess >= 3:
                severity = "severe"
            elif word.lower() in short_common_words and hyp_count <= 2:
                severity = "tolerable"  # Allow these to pass even if score is slightly lower
            elif hyp_count == 3 or excess == 2:
                severity = "moderate"
            else:
                severity = "minor"

            return {
                "is_hallucination": True,
                "pattern": word,
                "count": hyp_count,
                "ref_count": ref_count,
                "type": "single_word",
                "severity": severity
            }

    # Check for hallucinated phrase repetitions
    for phrase, hyp_count in hyp_phrase_repeats.items():
        ref_count = ref_phrase_repeats.get(phrase, 0)

        # Only flag if hypothesis has MORE repetitions than reference
        if hyp_count > ref_count:
            return {
                "is_hallucination": True,
                "pattern": phrase,
                "count": hyp_count,
                "ref_count": ref_count,
                "type": "phrase",
                "severity": "severe"  # Phrase repetition is always severe
            }

    return {"is_hallucination": False}

def detect_truncation(ref_text: str, hyp_text: str) -> dict:
    """
    Detect if ASR transcription seems truncated compared to reference.
    Returns: {"is_truncated": bool, "ref_words": int, "hyp_words": int, "ratio": float}
    """
    ref_words = len(ref_text.split())
    hyp_words = len(hyp_text.split())

    if ref_words == 0:
        return {"is_truncated": False, "ref_words": 0, "hyp_words": hyp_words, "ratio": 1.0}

    ratio = hyp_words / ref_words

    # Flag if transcribed < 40% of expected words
    is_truncated = ratio < 0.4

    return {
        "is_truncated": is_truncated,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
        "ratio": ratio
    }

# ============================================================================
# SINGLE CHUNK VALIDATION + FAST PIPELINE (Phase 1–2)
# ============================================================================

_ASR_TARGET_SR = 16000
_MIN_AUDIO_DURATION_S = 0.5


def load_audio_mono_16k(audio_path: Path | str) -> Tuple[np.ndarray, float]:
    """Load one WAV as mono float32 at 16 kHz for Whisper (single decode).

    Args:
        audio_path: Path to a chunk WAV.

    Returns:
        (samples, duration_seconds) at 16 kHz mono.

    Raises:
        ValueError: Empty or unreadable audio.
        OSError: File missing / IO errors from readers.
    """
    path = Path(audio_path)
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float32)
        if int(sr) != _ASR_TARGET_SR:
            data = librosa.resample(
                data, orig_sr=int(sr), target_sr=_ASR_TARGET_SR
            ).astype(np.float32)
    else:
        data, _sr = librosa.load(str(path), sr=_ASR_TARGET_SR, mono=True)
        data = np.asarray(data, dtype=np.float32)

    if data.size == 0:
        raise ValueError("Audio file has 0 frames")
    duration = float(data.shape[0]) / float(_ASR_TARGET_SR)
    return np.ascontiguousarray(data, dtype=np.float32), duration


def transcribe_audio(
    asr_model,
    audio: np.ndarray | str | Path,
    language: str = "en",
    vad_filter: bool = False,
) -> str:
    """Run Whisper on a 16 kHz array or path; return joined segment text.

    TTS chunks are short clean speech: default ``vad_filter=False`` avoids
    Silero VAD cost and edge chopping. Callers that need VAD can pass True.

    Args:
        asr_model: faster-whisper WhisperModel or compatible (``.transcribe``).
        audio: Mono float32 @ 16 kHz, or a filesystem path.
        language: Language pin (e.g. ``en``).
        vad_filter: Whether to run VAD before decode.

    Returns:
        Transcript string (may be empty).
    """
    segs = transcribe_segments(
        asr_model, audio, language=language, vad_filter=vad_filter
    )
    return " ".join(s["text"] for s in segs if s.get("text")).strip()


def transcribe_segments(
    asr_model,
    audio: np.ndarray | str | Path,
    language: str = "en",
    vad_filter: bool = False,
) -> List[Dict[str, Any]]:
    """Run Whisper and return timed segments ``{start, end, text}``.

    Used by Phase 3 packing to map speech back onto packed chunk windows.

    Args:
        asr_model: Model with ``.transcribe``.
        audio: Mono float32 @ 16 kHz or path.
        language: Language pin.
        vad_filter: Optional Silero VAD.

    Returns:
        List of segment dicts with start/end seconds and text.
    """
    kwargs: Dict[str, Any] = {
        "language": language or "en",
        "condition_on_previous_text": False,
        "vad_filter": bool(vad_filter),
        # Need real segment times to split packed multi-chunk audio
        "word_timestamps": False,
        "without_timestamps": False,
    }
    if vad_filter:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}

    source: Any = str(audio) if isinstance(audio, (str, Path)) else audio
    segments, _info = asr_model.transcribe(source, **kwargs)
    out: List[Dict[str, Any]] = []
    for seg in segments:
        text = getattr(seg, "text", None)
        if text is None:
            text = str(seg)
        text = (text or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(getattr(seg, "start", 0.0) or 0.0),
                "end": float(getattr(seg, "end", 0.0) or 0.0),
                "text": text,
            }
        )
    return out


def pack_audio_regions(
    items: List[Dict[str, Any]],
    silence_s: float = 0.75,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Concatenate chunk arrays with silence pads; return audio + time regions.

    Args:
        items: Dicts with keys ``chunk_num``, ``audio`` (float32 mono 16 kHz),
            ``ref_text``, ``audio_path``.
        silence_s: Silence inserted between chunks (helps boundary isolation).

    Returns:
        (packed_audio, regions) where each region has start_s/end_s and metadata.
    """
    silence_n = max(0, int(round(float(silence_s) * _ASR_TARGET_SR)))
    silence = (
        np.zeros(silence_n, dtype=np.float32) if silence_n > 0 else None
    )
    parts: List[np.ndarray] = []
    regions: List[Dict[str, Any]] = []
    t = 0.0
    for i, item in enumerate(items):
        if i > 0 and silence is not None:
            parts.append(silence)
            t += silence_n / float(_ASR_TARGET_SR)
        audio = np.ascontiguousarray(item["audio"], dtype=np.float32)
        start_s = t
        dur = float(audio.shape[0]) / float(_ASR_TARGET_SR)
        end_s = start_s + dur
        parts.append(audio)
        t = end_s
        regions.append(
            {
                "chunk_num": item["chunk_num"],
                "ref_text": item.get("ref_text") or "",
                "audio_path": item.get("audio_path") or "",
                "start_s": start_s,
                "end_s": end_s,
                "audio": audio,  # kept for solo retry if mapping empty
            }
        )
    if not parts:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(parts), regions


def assign_segments_to_regions(
    segments: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map Whisper segments onto packed chunk time windows by midpoint.

    Args:
        segments: Timed segments from ``transcribe_segments``.
        regions: Packed regions with ``start_s`` / ``end_s`` / ``chunk_num``.

    Returns:
        Map chunk_num → joined hypothesis text (missing keys → empty string).
    """
    hyps: Dict[str, List[str]] = {r["chunk_num"]: [] for r in regions}
    if not regions:
        return {}

    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        mid = 0.5 * (start + end)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        # Prefer region containing midpoint; else max time-overlap
        chosen = None
        for r in regions:
            if r["start_s"] - 1e-3 <= mid <= r["end_s"] + 1e-3:
                chosen = r["chunk_num"]
                break
        if chosen is None:
            best_overlap = -1.0
            best_id = regions[0]["chunk_num"]
            for r in regions:
                ov = min(end, r["end_s"]) - max(start, r["start_s"])
                if ov > best_overlap:
                    best_overlap = ov
                    best_id = r["chunk_num"]
            chosen = best_id
        hyps[chosen].append(text)

    return {k: " ".join(v).strip() for k, v in hyps.items()}


def _empty_result(chunk_num: str, audio_path: str = "") -> Dict[str, Any]:
    """Build a blank validation result dict for one chunk."""
    return {
        "chunk_num": chunk_num,
        "passed": False,
        "score": 0.0,
        "ref_text_raw": "",
        "ref_normalized": "",
        "hyp_text_raw": "",
        "hyp_normalized": "",
        "audio_path": audio_path,
        "error": "",
        "classification": "FAIL",
        "failure_type": None,
        "coverage_score": 0.0,
        "phonetic_score": 0.0,
        "extra_tokens": [],
        "missing_tokens": [],
        "substitutions": [],
        "accepted_equivalences": [],
        "accepted_phrase_equivalences": [],
        "accepted_ambiguous_equivalences": [],
        "accepted_list_label_equivalences": [],
        "accepted_book_term_equivalences": [],
        "critical_mismatches": [],
        "identifier_comparisons": [],
        "requires_second_stage_confirmation": False,
        "second_stage_confirmation_reason": "",
        "repetition_details": {},
        "alignment_operations": [],
        "component_scores": {},
        "explanation": "",
    }


def build_tts_book_term_evidence(tts_dir: Path | str) -> Dict[str, Any]:
    """Build one book-local recurring-term evidence set from original chunks.

    This reads source text only. It never records ASR spellings or writes a
    per-book alias file, so later comparison remains a bounded decision rather
    than a maintained mapping.

    Args:
        tts_dir: TTS directory containing ``text_chunks`` source files.

    Returns:
        JSON-safe recurring source-term evidence for one Stage 2 batch.
    """
    text_dir = Path(tts_dir) / "text_chunks"
    if not text_dir.is_dir():
        return build_book_term_evidence(())
    texts: List[str] = []
    for text_path in sorted(text_dir.glob("chunk_*.txt")):
        try:
            texts.append(text_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logging.warning("Could not read source text for book-term evidence %s: %s", text_path, exc)
    return build_book_term_evidence(texts)


def score_asr_pair(
    chunk_num: str,
    ref_text: str,
    hyp_text_raw: str,
    threshold: float,
    audio_path: str = "",
    book_term_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score hypothesis vs reference (CPU only — no model / no GPU).

    Phase 2: run this off the GPU thread so transcription never waits on
    normalize / fuzz / explain_diff.

    Args:
        chunk_num: Chunk id for logging and result payload.
        ref_text: Original reference text.
        hyp_text_raw: ASR transcript.
        threshold: Pass threshold.
        audio_path: Optional path stored in the result.
        book_term_evidence: Optional recurring-term evidence used only by Stage 2.

    Returns:
        Full validation result dict (same shape as validate_single_chunk).
    """
    result_template = _empty_result(chunk_num, audio_path)
    result_template["ref_text_raw"] = ref_text or ""
    result_template["hyp_text_raw"] = hyp_text_raw or ""

    if not ref_text:
        result_template["error"] = "Reference text is empty"
        return result_template

    ref_normalized, ref_id_contexts = normalize(ref_text)
    ref_id_keys = [ctx["canonical_id"] for ctx in ref_id_contexts]
    result_template["ref_id_keys"] = ref_id_keys

    hyp_normalized, hyp_id_contexts = normalize(hyp_text_raw or "")
    hyp_id_keys = [ctx["canonical_id"] for ctx in hyp_id_contexts]

    # The structured scorer preserves slot positions while still penalizing
    # ordinary missing or substituted speech outside those slots.
    structured = compare_spoken(
        ref_text,
        hyp_text_raw or "",
        threshold=threshold,
        book_term_evidence=book_term_evidence,
    )

    # Matching placeholders are wildcard slots. Preserve them in the compared
    # text so identifier handling cannot remove neighboring spoken prose.
    filtered_ref_text = structured["ref_normalized"]
    filtered_hyp_text = structured["hyp_normalized"]
    result_template["ref_normalized"] = filtered_ref_text
    result_template["hyp_normalized"] = filtered_hyp_text
    result_template["hyp_id_keys"] = hyp_id_keys

    hallucination_check = detect_hallucination(ref_normalized, hyp_normalized)
    truncation_check = detect_truncation(ref_normalized, hyp_normalized)

    if hallucination_check["is_hallucination"]:
        ref_count = hallucination_check.get("ref_count", 0)
        result_template["hallucination_warning"] = (
            f"Hallucination detected: '{hallucination_check['pattern']}' "
            f"repeated {hallucination_check['count']} times in hypothesis "
            f"(vs {ref_count} in reference, type: {hallucination_check['type']}, "
            f"severity: {hallucination_check.get('severity', 'unknown')})"
        )

    if truncation_check["is_truncated"]:
        result_template["truncation_warning"] = (
            f"Possible truncation: {truncation_check['hyp_words']} words "
            f"vs {truncation_check['ref_words']} expected "
            f"({truncation_check['ratio']:.1%})"
        )

    prose_score = float(structured["score"])

    id_score = float(structured.get("id_score") or 0.0)
    combined_score = prose_score
    minor_mismatch = bool(structured["minor_mismatch"])

    is_truncated = truncation_check["is_truncated"]
    is_hallucinated = hallucination_check["is_hallucination"]
    hallucination_severity = hallucination_check.get("severity", "unknown")
    is_severe_hallu = is_hallucinated and hallucination_severity == "severe"
    is_soft_hallu = is_hallucinated and hallucination_severity in (
        "minor",
        "tolerable",
        "moderate",
    )

    id_forgiveness = (
        id_score < 1.0
        and prose_score >= 0.95
        and combined_score < threshold
    )

    ref_tokens = filtered_ref_text.split() if filtered_ref_text else []
    hyp_tokens = filtered_hyp_text.split() if filtered_hyp_text else []
    alignment = {
        "operations": list(structured["alignment_operations"]),
        "extra_tokens": list(structured["extra_tokens"]),
        "missing_tokens": list(structured["missing_tokens"]),
        "substitutions": list(structured["substitutions"]),
        "coverage_score": float(structured["coverage_score"]),
        "phonetic_score": float(structured["phonetic_score"]),
    }

    extra_info = (
        detect_extra_speech(alignment)
        if "detect_extra_speech" in globals()
        else {"unexpected": []}
    )
    missing_info = (
        detect_missing_speech(alignment)
        if "detect_missing_speech" in globals()
        else {"missing": []}
    )
    rep_info = (
        detect_repetition(ref_tokens, hyp_tokens)
        if "detect_repetition" in globals()
        else {"has_repetition": False}
    )
    crit_info = (
        check_critical_tokens(alignment)
        if "check_critical_tokens" in globals()
        else {"critical_fail": False}
    )

    has_unexpected = bool(extra_info.get("unexpected"))
    has_missing_critical = bool(missing_info.get("missing")) or crit_info.get(
        "critical_fail", False
    )
    has_severe_rep = (
        rep_info.get("has_repetition", False)
        and rep_info.get("type") != "expected_ref_repeat"
    )

    if is_truncated or has_severe_rep or has_unexpected or has_missing_critical:
        classification = "FAIL"
        passed = False
        if has_unexpected:
            result_template["failure_type"] = "unexpected_extra_speech"
        elif has_missing_critical:
            result_template["failure_type"] = "missing_or_critical"
        elif has_severe_rep:
            result_template["failure_type"] = "unexpected_repetition"
        else:
            result_template["failure_type"] = "truncated"
    elif is_severe_hallu:
        classification = "FAIL"
        passed = False
        result_template["failure_type"] = "severe_hallucination"
    elif is_soft_hallu or minor_mismatch or id_forgiveness:
        # ASR quality control is binary: soft differences pass only when they
        # clear the configured score threshold; otherwise they are failures.
        classification = "PASS" if combined_score >= threshold else "FAIL"
        passed = classification == "PASS"
    else:
        classification = "PASS" if combined_score >= threshold else "FAIL"
        passed = classification == "PASS"

    # The shared comparator is the one scoring authority. Legacy checks above
    # remain only for legacy warning fields until their callers are retired.
    passed = bool(structured["passed"])
    classification = str(structured["classification"])
    combined_score = float(structured["score"])
    minor_mismatch = bool(structured["minor_mismatch"])
    result_template.update(
        {
            "passed": passed,
            "prose_score": prose_score,
            "id_score": id_score,
            "score": combined_score,
            "minor_mismatch": minor_mismatch,
            "classification": classification,
            "failure_type": structured.get("failure_type"),
            "coverage_score": structured["coverage_score"],
            "phonetic_score": structured["phonetic_score"],
            "extra_tokens": list(structured["extra_tokens"]),
            "missing_tokens": list(structured["missing_tokens"]),
            "substitutions": list(structured["substitutions"]),
            "accepted_equivalences": list(structured["accepted_equivalences"]),
            "accepted_phrase_equivalences": list(
                structured.get("accepted_phrase_equivalences", [])
            ),
            "accepted_ambiguous_equivalences": list(
                structured.get("accepted_ambiguous_equivalences", [])
            ),
            "accepted_list_label_equivalences": list(structured.get("accepted_list_label_equivalences", [])),
            "accepted_book_term_equivalences": list(
                structured.get("accepted_book_term_equivalences", [])
            ),
            "critical_mismatches": list(structured["critical_mismatches"]),
            "identifier_comparisons": list(structured.get("identifier_comparisons", [])),
            "requires_second_stage_confirmation": bool(
                structured.get("requires_second_stage_confirmation")
            ),
            "second_stage_confirmation_reason": structured.get(
                "second_stage_confirmation_reason", ""
            ),
            "repetition_details": list(structured["repetition_details"]),
            "alignment_operations": list(structured["alignment_operations"]),
            "error": "" if passed else result_template.get("error", ""),
        }
    )

    if not passed:
        if is_truncated:
            result_template["explanation"] = result_template.get(
                "truncation_warning", ""
            )
        elif is_hallucinated:
            result_template["explanation"] = result_template.get(
                "hallucination_warning", ""
            )
        else:
            result_template["explanation"] = structured["explanation"]

    return result_template


def validate_single_chunk(
    chunk_num: str,
    tts_dir: Path,
    threshold: float,
    asr_model,
    pass_tolerance_score: float = PASS_TOLERANCE_SCORE,
    audio_path_override: Path | None = None,
    language: str = "en",
    audio_array: np.ndarray | None = None,
    vad_filter: bool = False,
) -> Dict[str, Any]:
    """Validate a single chunk: one audio load → ASR → score.

    Args:
        chunk_num: Chunk stem like chunk_00005.
        tts_dir: Parent TTS directory with audio_chunks and text_chunks.
        threshold: Minimum prose similarity score to pass.
        asr_model: Loaded ASR model with ``.transcribe``.
        pass_tolerance_score: Deprecated; kept for call-site compat.
        audio_path_override: Optional explicit WAV path.
        language: Whisper language code (e.g. en).
        audio_array: Optional preloaded mono float32 @ 16 kHz (skips disk load).
        vad_filter: Silero VAD before decode (default False for TTS speed).
    """
    _ = pass_tolerance_score
    audio_path = (
        audio_path_override
        if audio_path_override
        else Path(tts_dir) / "audio_chunks" / f"{chunk_num}.wav"
    )
    text_path = Path(tts_dir) / "text_chunks" / f"{chunk_num}.txt"
    result_template = _empty_result(chunk_num, str(audio_path))

    if audio_array is None:
        if not Path(audio_path).is_file():
            result_template["error"] = "Audio file does not exist"
            return result_template
        if os.path.getsize(audio_path) == 0:
            result_template["error"] = "Audio file is empty (0 bytes)"
            return result_template
        try:
            audio_data, duration_seconds = load_audio_mono_16k(audio_path)
        except Exception as e:
            logging.error(
                "Skipping %s: Failed to load audio file %s: %s",
                chunk_num,
                audio_path,
                e,
            )
            result_template["error"] = f"Failed to load audio file: {e}"
            return result_template
    else:
        audio_data = np.ascontiguousarray(audio_array, dtype=np.float32)
        duration_seconds = float(audio_data.shape[0]) / float(_ASR_TARGET_SR)

    if duration_seconds < _MIN_AUDIO_DURATION_S:
        result_template["error"] = (
            f"Audio file too short ({duration_seconds:.2f}s < {_MIN_AUDIO_DURATION_S}s)"
        )
        return result_template

    if not text_path.is_file():
        result_template["error"] = "Text file does not exist"
        return result_template
    if os.path.getsize(text_path) == 0:
        result_template["error"] = "Text file is empty (0 bytes)"
        return result_template

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    except Exception as e:
        result_template["error"] = f"Failed to read text file: {e}"
        return result_template

    if not ref_text:
        result_template["error"] = "Reference text is empty"
        return result_template

    try:
        hyp_text_raw = transcribe_audio(
            asr_model,
            audio_data,
            language=language or "en",
            vad_filter=vad_filter,
        )
    except Exception as e:
        logging.error(
            "ASR transcription failed for %s (%s): %s", chunk_num, audio_path, e
        )
        result_template["error"] = f"ASR transcription error: {e}"
        return result_template

    result = score_asr_pair(
        chunk_num,
        ref_text,
        hyp_text_raw,
        threshold,
        audio_path=str(audio_path),
    )
    logging.debug(
        "ASR validation %s: score=%.3f passed=%s class=%s",
        chunk_num,
        float(result.get("score") or 0),
        result.get("passed"),
        result.get("classification"),
    )
    return result


def run_pipeline_batch_validation(
    tts_dir: Path,
    chunks: List[str],
    asr_model,
    threshold: float,
    language: str = "en",
    load_workers: int = 4,
    score_workers: int = 4,
    vad_filter: bool = False,
    pack_size: int = 16,
    pack_silence_s: float = 0.75,
    max_pack_seconds: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
    progress_cb: Optional[Any] = None,
    book_term_evidence: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Fast batch: CPU preload → packed GPU decode → parallel CPU score.

    Phase 1: one model; loaders decode WAV→16 kHz once; no VAD by default.
    Phase 2: scoring on a thread pool so GPU never waits on fuzz.
    Phase 3: pack ``pack_size`` clips into one stream with silence pads, one
    Whisper pass, map segments back by time; solo retry if a window is empty.

    Args:
        tts_dir: TTS dir with audio_chunks / text_chunks.
        chunks: Chunk stems to validate.
        asr_model: Single loaded ASR model (use on one thread only).
        threshold: Pass threshold.
        language: Whisper language pin.
        load_workers: Parallel disk/decode threads.
        score_workers: Parallel scoring threads.
        vad_filter: Pass True only if VAD is required.
        pack_size: Chunks per GPU pack (1 = no packing).
        pack_silence_s: Silence between packed clips (seconds).
        max_pack_seconds: Optional cap on summed clip duration per pack. When
            omitted, keep legacy pack formation behavior.
        stop_event: Optional cooperative cancel.
        progress_cb: Optional ``cb(done, total, result)`` after each score.
        book_term_evidence: Recurring source-term evidence for Stage 2 only.

    Returns:
        List of result dicts (order not guaranteed).
    """
    tts_dir = Path(tts_dir)
    total = len(chunks)
    if total == 0:
        return []

    load_workers = max(1, int(load_workers))
    score_workers = max(1, int(score_workers))
    pack_size = max(1, int(pack_size))
    pack_silence_s = max(0.0, float(pack_silence_s))
    max_pack_seconds = (
        None if max_pack_seconds is None else max(0.0, float(max_pack_seconds))
    )
    # Bound queues so loaders cannot blow RAM with thousands of arrays
    load_q: queue.Queue = queue.Queue(maxsize=max(pack_size * 2, load_workers * 2, 8))
    score_q: queue.Queue = queue.Queue(maxsize=max(8, score_workers * 2))
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    done_count = 0
    stop = stop_event or threading.Event()

    def _stopped() -> bool:
        """Check if a stop signal has been set."""
        return stop.is_set()

    def _emit_score(
        chunk_num: str,
        ref_text: str,
        hyp_text: str,
        audio_path: str,
        error: Optional[str] = None,
    ) -> None:
        """Push one scored item to the score queue."""
        score_q.put(
            {
                "chunk_num": chunk_num,
                "ref_text": ref_text or "",
                "hyp_text": hyp_text or "",
                "audio_path": audio_path or "",
                "error": error,
            }
        )

    def _safe_score_asr_pair(
        chunk_num: str,
        ref_text: str,
        hyp_text: str,
        audio_path: str,
    ) -> Dict[str, Any]:
        """Score one transcript without allowing a comparator error to kill a worker."""
        try:
            return score_asr_pair(
                chunk_num,
                ref_text,
                hyp_text,
                threshold,
                audio_path=audio_path,
                book_term_evidence=book_term_evidence,
            )
        except Exception as exc:
            logging.exception("ASR comparison failed for %s", chunk_num)
            result = _empty_result(chunk_num, audio_path)
            result.update(
                {
                    "error": f"ASR comparison error: {exc}",
                    "ref_text_raw": ref_text or "",
                    "hyp_text_raw": hyp_text or "",
                }
            )
            return result

    def _flush_pack(pack: List[Dict[str, Any]]) -> None:
        """Transcribe a packed batch (or solo) and enqueue hyps for scoring."""
        if not pack:
            return
        # Single item: no concat overhead
        if len(pack) == 1:
            item = pack[0]
            try:
                hyp = transcribe_audio(
                    asr_model,
                    item["audio"],
                    language=language or "en",
                    vad_filter=vad_filter,
                )
                _emit_score(
                    item["chunk_num"],
                    item.get("ref_text") or "",
                    hyp,
                    item.get("audio_path") or "",
                )
            except Exception as exc:
                _emit_score(
                    item["chunk_num"],
                    item.get("ref_text") or "",
                    "",
                    item.get("audio_path") or "",
                    error=f"ASR transcription error: {exc}",
                )
            item["audio"] = None
            return

        try:
            packed, regions = pack_audio_regions(pack, silence_s=pack_silence_s)
            segments = transcribe_segments(
                asr_model,
                packed,
                language=language or "en",
                vad_filter=vad_filter,
            )
            hyp_map = assign_segments_to_regions(segments, regions)
        except Exception as exc:
            # Whole pack failed — solo each clip so we still finish the book
            logging.warning("Pack transcribe failed (%s); solo fallback ×%s", exc, len(pack))
            for item in pack:
                try:
                    hyp = transcribe_audio(
                        asr_model,
                        item["audio"],
                        language=language or "en",
                        vad_filter=vad_filter,
                    )
                    _emit_score(
                        item["chunk_num"],
                        item.get("ref_text") or "",
                        hyp,
                        item.get("audio_path") or "",
                    )
                except Exception as solo_exc:
                    _emit_score(
                        item["chunk_num"],
                        item.get("ref_text") or "",
                        "",
                        item.get("audio_path") or "",
                        error=f"ASR transcription error: {solo_exc}",
                    )
                item["audio"] = None
            return

        # Accuracy gate: packed mapping often mis-assigns. Keep packed hyp only
        # when it already passes threshold; otherwise solo re-decode that clip
        # (same path as Phase 1 — restores quality without packing the whole book).
        solo_retries = 0
        for r in regions:
            chunk_num = r["chunk_num"]
            hyp = hyp_map.get(chunk_num, "")
            ref = r.get("ref_text") or ""
            path = r.get("audio_path") or ""
            audio_solo = r.get("audio")
            need_solo = False
            if not hyp and ref.strip():
                need_solo = True
            elif ref.strip() and hyp:
                prelim = _safe_score_asr_pair(chunk_num, ref, hyp, path)
                if not prelim.get("error") and not prelim.get("passed"):
                    need_solo = True
            if need_solo and audio_solo is not None:
                try:
                    hyp2 = transcribe_audio(
                        asr_model,
                        audio_solo,
                        language=language or "en",
                        vad_filter=vad_filter,
                    )
                    solo_retries += 1
                    # Prefer solo if it scores better or packed was empty
                    if not hyp:
                        hyp = hyp2
                    else:
                        s_pack = _safe_score_asr_pair(chunk_num, ref, hyp, path)
                        s_solo = _safe_score_asr_pair(chunk_num, ref, hyp2, path)
                        if float(s_solo.get("score") or 0) >= float(
                            s_pack.get("score") or 0
                        ):
                            hyp = hyp2
                except Exception as solo_exc:
                    if not hyp:
                        _emit_score(
                            chunk_num,
                            ref,
                            "",
                            path,
                            error=f"ASR transcription error: {solo_exc}",
                        )
                        r["audio"] = None
                        continue
            _emit_score(chunk_num, ref, hyp, path)
            r["audio"] = None
        if solo_retries:
            logging.info(
                "Pack accuracy gate: solo-retried %s / %s clips",
                solo_retries,
                len(regions),
            )
        for item in pack:
            item["audio"] = None

    def loader_loop(chunk_slice: List[str]) -> None:
        """Load audio + reference text for assigned chunks into load_q."""
        for chunk_num in chunk_slice:
            if _stopped():
                break
            audio_path = tts_dir / "audio_chunks" / f"{chunk_num}.wav"
            text_path = tts_dir / "text_chunks" / f"{chunk_num}.txt"
            try:
                if not audio_path.is_file():
                    raise FileNotFoundError("Audio file does not exist")
                if os.path.getsize(audio_path) == 0:
                    raise ValueError("Audio file is empty (0 bytes)")
                audio_data, duration = load_audio_mono_16k(audio_path)
                if duration < _MIN_AUDIO_DURATION_S:
                    raise ValueError(
                        f"Audio file too short ({duration:.2f}s < {_MIN_AUDIO_DURATION_S}s)"
                    )
                if not text_path.is_file():
                    raise FileNotFoundError("Text file does not exist")
                ref_text = text_path.read_text(encoding="utf-8").strip()
                if not ref_text:
                    raise ValueError("Reference text is empty")
                load_q.put(
                    {
                        "chunk_num": chunk_num,
                        "audio": audio_data,
                        "duration_s": duration,
                        "ref_text": ref_text,
                        "audio_path": str(audio_path),
                        "error": None,
                    }
                )
            except Exception as exc:
                load_q.put(
                    {
                        "chunk_num": chunk_num,
                        "audio": None,
                        "duration_s": 0.0,
                        "ref_text": "",
                        "audio_path": str(audio_path),
                        "error": str(exc),
                    }
                )
        load_q.put(None)  # per-loader sentinel

    gpu_errors: List[BaseException] = []

    def gpu_loop() -> None:
        """Pack preloaded clips and run fewer GPU passes (only thread using model)."""
        finished_loaders = 0
        pack: List[Dict[str, Any]] = []
        pack_duration_s = 0.0
        try:
            while finished_loaders < load_workers:
                if _stopped() and load_q.empty() and not pack:
                    break
                try:
                    item = load_q.get(timeout=0.2)
                except queue.Empty:
                    # Timeout with a partial pack: flush so GPU stays busy
                    if pack:
                        _flush_pack(pack)
                        pack = []
                        pack_duration_s = 0.0
                    continue
                if item is None:
                    finished_loaders += 1
                    continue
                if item.get("error"):
                    _emit_score(
                        item["chunk_num"],
                        item.get("ref_text") or "",
                        "",
                        item.get("audio_path") or "",
                        error=item["error"],
                    )
                    continue
                item_duration_s = float(item.get("duration_s") or 0.0)
                if (
                    pack
                    and max_pack_seconds is not None
                    and (pack_duration_s + item_duration_s) > max_pack_seconds
                ):
                    # Duration cap applies before appending the next loaded clip
                    # so one long item cannot silently overflow an established pack.
                    _flush_pack(pack)
                    pack = []
                    pack_duration_s = 0.0
                pack.append(item)
                pack_duration_s += item_duration_s
                if len(pack) >= pack_size or (
                    max_pack_seconds is not None and pack_duration_s >= max_pack_seconds
                ):
                    _flush_pack(pack)
                    pack = []
                    pack_duration_s = 0.0
            if pack:
                _flush_pack(pack)
        except Exception as exc:
            gpu_errors.append(exc)
            stop.set()
            logging.exception("ASR GPU batch loop failed")
            # Loader threads may be blocked on the bounded queue after a GPU
            # failure; drain it until they observe the stop event and exit.
            while any(loader.is_alive() for loader in loaders):
                try:
                    load_q.get(timeout=0.1)
                except queue.Empty:
                    pass
        finally:
            # Always release scorer threads, including unexpected GPU failures.
            for _ in range(score_workers):
                score_q.put(None)

    def scorer_loop() -> None:
        """CPU score transcripts; invoke progress_cb."""
        nonlocal done_count
        while True:
            item = score_q.get()
            if item is None:
                break
            chunk_num = item["chunk_num"]
            if item.get("error"):
                result = _empty_result(chunk_num, item.get("audio_path") or "")
                result["error"] = item["error"]
                result["ref_text_raw"] = item.get("ref_text") or ""
            else:
                result = _safe_score_asr_pair(
                    chunk_num,
                    item.get("ref_text") or "",
                    item.get("hyp_text") or "",
                    item.get("audio_path") or "",
                )
            with results_lock:
                results.append(result)
                done_count += 1
                cur = done_count
            if progress_cb is not None:
                try:
                    progress_cb(cur, total, result)
                except Exception:
                    pass

    # Partition chunks across loaders
    slices: List[List[str]] = [[] for _ in range(load_workers)]
    for i, c in enumerate(chunks):
        slices[i % load_workers].append(c)

    loaders = [
        threading.Thread(
            target=loader_loop, args=(slices[i],), name=f"asr-load-{i}", daemon=True
        )
        for i in range(load_workers)
    ]
    scorers = [
        threading.Thread(
            target=scorer_loop, name=f"asr-score-{i}", daemon=True
        )
        for i in range(score_workers)
    ]
    gpu_thread = threading.Thread(target=gpu_loop, name="asr-gpu", daemon=True)

    for t in loaders:
        t.start()
    for t in scorers:
        t.start()
    gpu_thread.start()

    for t in loaders:
        t.join()
    gpu_thread.join()
    for t in scorers:
        t.join()

    if gpu_errors:
        raise RuntimeError("ASR GPU batch loop failed") from gpu_errors[0]

    return results

# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_batch(tts_dir: Path, threshold: float, progress_queue: queue.Queue,
                   max_workers: int = 1, model_name: str = DEFAULT_ASR_MODEL) -> Dict[str, Any]:
    """
    Validate all chunks in parallel.
    Uses faster-whisper for ASR transcription.
    """
    progress_queue.put({"type": "status", "message": "Discovering chunks..."})
    chunk_nums = discover_chunks(tts_dir)

    if not chunk_nums:
        progress_queue.put({"type": "status", "message": "No matching chunk pairs found."})
        return {"error": "No matching chunk pairs found", "total": 0, "passed": 0, "failed": 0, "results": [], "failed_chunks": []}

    total_chunks = len(chunk_nums)
    progress_queue.put({"type": "status", "message": f"Found {total_chunks} chunks. Loading ASR model..."})

    # Load ASR model
    asr_model, device = load_asr_model_adaptive(model_name)
    if not asr_model:
        progress_queue.put({"type": "status", "message": "Failed to load ASR model"})
        return {"error": "Failed to load ASR model", "total": 0, "passed": 0, "failed": 0, "results": [], "failed_chunks": []}

    progress_queue.put({"type": "status", "message": f"ASR model loaded on {_format_device(device)}. Starting validation..."})

    results = []
    all_passed = []
    all_failed = []

    def validate_wrapper(chunk_num):
        """Runs validation on multiple chunks concurrently using ThreadPoolExecutor."""
        return validate_single_chunk(chunk_num, tts_dir, threshold, asr_model)

    # Start timing
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(validate_wrapper, chunk_num) for chunk_num in chunk_nums]

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)

                    if result["passed"]:
                        all_passed.append(result)
                    else:
                        all_failed.append(result)

                    # Calculate timing metrics
                    elapsed_time = time.time() - start_time
                    chunks_remaining = total_chunks - i

                    # Calculate ETA based on average time per chunk
                    if i > 0:
                        avg_time_per_chunk = elapsed_time / i
                        eta = avg_time_per_chunk * chunks_remaining
                    else:
                        eta = 0

                    # Format times
                    elapsed_str = format_time(elapsed_time)
                    eta_str = format_time(eta)

                    # Update progress
                    progress_queue.put({
                        "type": "progress",
                        "current": i,
                        "total": total_chunks
                    })
                    progress_queue.put({
                        "type": "status",
                        "message": f"Validated {i} of {total_chunks} chunks | Elapsed: {elapsed_str} | ETA: {eta_str}"
                    })

                except Exception as e:
                    logging.error(f"Validation error: {e}")

    finally:
        # Cleanup ASR model
        cleanup_asr_model(asr_model)

    return {
        "total": len(results),
        "passed": len(all_passed),
        "failed": len(all_failed),
        "results": results,
        "failed_chunks": all_failed,
    }

# ============================================================================
# TKINTER GUI
# ============================================================================

class ASRApp:
    """ASRApp: Initializes and manages the graphical user interface for an ASR validation tool using faster-whisper. Sets up the window title, size, loads the last used folder, initializes a queue for thread-safe updates, creates necessary widgets, and sets up logging redirection. Starts monitoring the queue for updates."""
    def __init__(self, root):
        """Initialize the ASR Validation Tool GUI.
        Args:
        root (tk.Tk): The main window of the application.
        Returns: None
        """
        self.root = root
        self.root.title("ASR Validation Tool (faster-whisper)")
        self.root.geometry("800x600")

        # Load last used folder from config
        self.tts_folder_path = load_last_folder()
        self.progress_queue = queue.Queue()  # For thread-safe updates

        self.create_widgets()
        self.setup_logging_redirect()
        self.process_queue()  # Start checking for queue updates

    def create_widgets(self):
        """Creates and configures a set of widgets for selecting TTS mode and folder. Args: none Returns: none"""
        # Mode selector frame
        mode_frame = ttk.Frame(self.root, padding="5")
        mode_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=5)
        self.mode_var = tk.StringVar(value="Batch Mode")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                        values=["Batch Mode", "Chunk Mode"], state="readonly", width=15)
        self.mode_combo.pack(side="left", padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        ttk.Label(mode_frame, text="ASR Model:").pack(side="left", padx=10)
        self.model_var = tk.StringVar(value=self._load_asr_model())
        self.model_combo = ttk.Combobox(mode_frame, textvariable=self.model_var,
                                        values=["base", "small", "medium"],
                                        state="readonly", width=12)
        self.model_combo.pack(side="left", padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Frame for folder selection
        self.folder_frame = ttk.LabelFrame(self.root, text="TTS Folder Selection", padding="10")
        self.folder_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(self.folder_frame, text="Selected Folder:").grid(row=0, column=0, sticky="w", pady=2)
        self.tts_folder_entry = ttk.Entry(self.folder_frame, width=60, state='readonly')
        self.tts_folder_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.browse_button = ttk.Button(self.folder_frame, text="Browse...", command=self.browse_tts_folder)
        self.browse_button.grid(row=0, column=2, padx=5, pady=2)
        self.folder_frame.grid_columnconfigure(1, weight=1)

        # Set the entry field with the loaded folder path
        self.tts_folder_entry.config(state='normal')
        self.tts_folder_entry.insert(0, str(self.tts_folder_path))
        self.tts_folder_entry.config(state='readonly')

        # Chunk list frame (for Chunk Mode)
        self.chunk_frame = ttk.LabelFrame(self.root, text="Select Chunk", padding="10")
        # Don't pack yet - will show/hide based on mode

        chunk_list_frame = ttk.Frame(self.chunk_frame)
        chunk_list_frame.pack(fill="both", expand=True)

        self.chunk_listbox = tk.Listbox(chunk_list_frame, height=8, width=50, exportselection=False)
        self.chunk_listbox.pack(side="left", fill="both", expand=True)

        chunk_scrollbar = ttk.Scrollbar(chunk_list_frame, orient="vertical", command=self.chunk_listbox.yview)
        chunk_scrollbar.pack(side="right", fill="y")
        self.chunk_listbox.config(yscrollcommand=chunk_scrollbar.set)

        self.test_chunk_button = ttk.Button(self.chunk_frame, text="Test Selected Chunk",
                                             command=self.test_selected_chunk, state='disabled')
        self.test_chunk_button.pack(pady=5)

        # Single chunk result frame
        self.chunk_result_frame = ttk.LabelFrame(self.root, text="Result", padding="10")
        # Don't pack yet

        self.chunk_result_text = tk.Text(self.chunk_result_frame, height=6, width=70, wrap='word', state='disabled')
        self.chunk_result_text.pack(fill="both", expand=True)

        # Frame for similarity threshold
        self.threshold_frame = ttk.LabelFrame(self.root, text="Validation Settings", padding="10")
        self.threshold_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(self.threshold_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.threshold_var = tk.DoubleVar(value=0.75)  # Default threshold
        self.threshold_spinbox = ttk.Spinbox(self.threshold_frame, from_=0.0, to=1.0, increment=0.05,
                                              textvariable=self.threshold_var, width=8, format="%.2f")
        self.threshold_spinbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(self.threshold_frame, text="(0.0 - 1.0)").grid(row=0, column=2, padx=5, pady=2)

        # Run Validation Button (for Batch Mode)
        self.run_button = ttk.Button(self.root, text="Run Validation", command=self.start_validation_thread, state='disabled')
        self.run_button.pack(padx=10, pady=10, fill="x")

        # Status and Log Area
        self.status_frame = ttk.LabelFrame(self.root, text="Status and Output Log", padding="10")
        self.status_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.status_label = ttk.Label(self.status_frame, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", pady=2)

        # Device indicator label
        self.device_label = ttk.Label(self.status_frame, text="Device: N/A", relief="sunken", anchor="e")
        self.device_label.pack(fill="x", pady=2)

        self.progress_bar = ttk.Progressbar(self.status_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(fill="x", pady=2)
        self.progress_bar.stop()  # Hide initially

        self.log_text_widget = tk.Text(
            self.status_frame,
            height=15,
            width=80,
            state='disabled',
            wrap='word',
            bg='blue',              # Blue background
            fg='lime',              # Bright green text
            insertbackground='lime' # Bright green cursor
        )
        self.log_text_widget.pack(fill="both", expand=True, padx=2, pady=2)

        self.log_scrollbar = ttk.Scrollbar(self.status_frame, command=self.log_text_widget.yview)
        self.log_scrollbar.pack(side="right", fill="y")
        self.log_text_widget['yscrollcommand'] = self.log_scrollbar.set

        # Text widget tags for colored logging (adjusted for dark background)
        self.log_text_widget.tag_config('info', foreground='lime')      # Bright green for info
        self.log_text_widget.tag_config('warning', foreground='yellow') # Yellow for warnings
        self.log_text_widget.tag_config('error', foreground='red')      # Red for errors
        self.log_text_widget.tag_config('success', foreground='cyan')   # Cyan for success

    def on_mode_change(self, event=None):
        """Handle mode switch between Batch Mode and Chunk Mode."""
        mode = self.mode_var.get()
        if mode == "Chunk Mode":
            # Show chunk selection widgets (pack after folder frame)
            self.chunk_frame.pack(padx=10, pady=5, fill="both", expand=False, after=self.folder_frame)
            self.chunk_result_frame.pack(padx=10, pady=5, fill="x", after=self.chunk_frame)
            # Hide batch run button
            self.run_button.pack_forget()
            # Populate chunk list if folder already selected
            if self.tts_folder_path:
                self.populate_chunk_list()
        else:
            # Batch Mode - hide chunk widgets
            self.chunk_frame.pack_forget()
            self.chunk_result_frame.pack_forget()
            # Show batch run button (pack after threshold frame)
            self.run_button.pack(padx=10, pady=10, fill="x", after=self.threshold_frame)
            if self.tts_folder_path:
                self.run_button.config(state='normal')

    def _on_model_change(self, event=None):
        """Persist model selection when changed."""
        _save_asr_model(self.model_var.get())

    def populate_chunk_list(self):
        """Populate the chunk listbox with available chunks from the TTS folder."""
        self.chunk_listbox.delete(0, tk.END)
        self.test_chunk_button.config(state='disabled')

        if not self.tts_folder_path:
            return

        chunks = discover_chunks(self.tts_folder_path)
        if chunks:
            for chunk in chunks:
                self.chunk_listbox.insert(tk.END, chunk)
            self.test_chunk_button.config(state='normal')
            self.update_log(f"Found {len(chunks)} chunks in folder.", 'info')
        else:
            self.update_log("No matching audio/text chunk pairs found.", 'warning')

    def test_selected_chunk(self):
        """Test the selected chunk from the listbox."""
        selection = self.chunk_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a chunk to test.")
            return

        chunk_num = self.chunk_listbox.get(selection[0])
        self.test_chunk_button.config(state='disabled')
        self.status_label.config(text=f"Testing {chunk_num}...")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()

        # Run in background thread
        test_thread = threading.Thread(target=self._run_single_chunk_test, args=(chunk_num,))
        test_thread.daemon = True
        test_thread.start()

    def _run_single_chunk_test(self, chunk_num: str):
        """Background thread to run single chunk validation."""
        try:
            self.progress_queue.put({"type": "status", "message": "Loading ASR model..."})
            model_name = self.model_var.get()
            asr_model, device = load_asr_model_adaptive(model_name)

            if not asr_model:
                self.progress_queue.put({"type": "chunk_result", "success": False,
                                         "message": "Failed to load ASR model"})
                return

            self.progress_queue.put({"type": "status", "message": f"ASR model loaded on {_format_device(device)}."})

            try:
                self.progress_queue.put({"type": "status", "message": f"Transcribing {chunk_num}..."})
                threshold = self.threshold_var.get()

                result = validate_single_chunk(chunk_num, self.tts_folder_path, threshold, asr_model)

                # Format result for display
                status = "PASSED" if result["passed"] else "FAILED"
                result_text = f"Chunk: {chunk_num}\n"
                result_text += f"Score: {result['score']:.2f}  Status: {status}\n\n"
                result_text += f"Reference: {result['ref_normalized']}\n\n"
                result_text += f"Transcribed: {result['hyp_normalized']}"

                if result.get("error"):
                    result_text += f"\n\nError: {result['error']}"

                # Write to log file
                log_path = self.tts_folder_path / "chunk_test.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(result_text)
                    f.write(f"\n{'='*60}\n")

                self.progress_queue.put({"type": "chunk_result", "success": True,
                                         "message": result_text, "passed": result["passed"]})
            finally:
                cleanup_asr_model(asr_model)

        except Exception as e:
            logging.exception(f"Single chunk test failed: {e}")
            self.progress_queue.put({"type": "chunk_result", "success": False,
                                     "message": f"Error: {e}"})

    def browse_tts_folder(self):
        """Opens a file dialog for selecting the TTS output folder. Updates the selected path and config accordingly.
        Args:
        None
        Returns:
        None
        """
        folder_path = filedialog.askdirectory(parent=self.root, title="Select TTS Output Folder")
        if folder_path:
            selected_path = Path(folder_path)

            # Auto-correct if user selected a subfolder
            if selected_path.name in ("text_chunks", "audio_chunks"):
                selected_path = selected_path.parent
                self.update_log(f"Auto-selected parent TTS folder: {selected_path}", 'info')

            self.tts_folder_path = selected_path
            # Save the selected folder to config
            save_last_folder(self.tts_folder_path)
            self.tts_folder_entry.config(state='normal')
            self.tts_folder_entry.delete(0, tk.END)
            self.tts_folder_entry.insert(0, str(self.tts_folder_path))
            self.tts_folder_entry.config(state='readonly')
            self.run_button.config(state='normal')
            self.update_log(f"Selected TTS Folder: {self.tts_folder_path}", 'info')

            # If in Chunk Mode, populate the chunk list
            if self.mode_var.get() == "Chunk Mode":
                self.populate_chunk_list()
        else:
            self.run_button.config(state='disabled')
            self.update_log("No folder selected.", 'warning')

    def setup_logging_redirect(self):
        """Sets up a custom logging handler to redirect logs to a Tkinter Text widget.
        Args:
        self (object): The instance of the class containing log_text_widget and progress_queue attributes.
        Returns:
        None
        """
        # Custom handler to redirect logging to the Tkinter Text widget
        class TextWidgetHandler(logging.Handler):
            """A class that extends `logging.Handler` to log messages into a text widget using a queue for asynchronous updates. The handler formats log records with timestamps, severity levels, and messages before passing them along with appropriate tags to the text widget."""
            def __init__(self, text_widget, queue):
                """Initializes a custom log handler for text widgets.
                Args:
                text_widget: The widget where logs will be displayed.
                queue: A queue to store log messages for further processing.
                Returns:
                None
                """
                super().__init__()
                self.text_widget = text_widget
                self.queue = queue
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            def emit(self, record):
                """Emit a log message to the widget.
                Args:
                record (logging.LogRecord): The log record to emit.
                Returns: None
                """
                msg = self.format(record)
                tag = record.levelname.lower()
                self.queue.put({"type": "log", "message": msg, "tag": tag})

        self.log_handler = TextWidgetHandler(self.log_text_widget, self.progress_queue)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)  # Ensure logger captures info and above

    def update_log(self, message, tag='info'):
        """Updates the log widget with a message and tag.
        Args:
        message (str): The message to log.
        tag (str, optional): Tag for the message style. Defaults to 'info'.
        Returns: None
        """
        self.log_text_widget.config(state='normal')
        self.log_text_widget.insert(tk.END, message + '\n', tag)
        self.log_text_widget.yview(tk.END)
        self.log_text_widget.config(state='disabled')

    def process_queue(self):
        """Handles the queue by processing items until it's empty. Updates the status and log accordingly. Colors the device indicator based on the type of device used.
        Args:
        None
        Returns:
        None
        """
        while not self.progress_queue.empty():
            item = self.progress_queue.get_nowait()
            if item["type"] == "status":
                self.status_label.config(text=item["message"])
                self.update_log(item["message"], 'info')
                # Check for device information in status messages
                if "ASR model loaded on" in item["message"]:
                    device = item["message"].split("ASR model loaded on ")[1].upper()
                    self.device_label.config(text=f"Device: {device}")
                    # Color code the device indicator
                    if device == "GPU":
                        self.device_label.config(foreground="green")
                    elif device == "CPU":
                        self.device_label.config(foreground="orange")
                    else:
                        self.device_label.config(foreground="black")
            elif item["type"] == "progress":
                if item["total"] > 0:
                    self.progress_bar.config(mode="determinate", maximum=item["total"], value=item["current"])
                    self.progress_bar.start()  # Ensure it's visible and moving
                else:
                    self.progress_bar.stop()
            elif item["type"] == "log":
                self.update_log(item["message"], item["tag"])
            elif item["type"] == "finished":
                self.run_button.config(state='normal')
                self.browse_button.config(state='normal')
                self.progress_bar.stop()
                self.progress_bar.config(value=0)
                self.status_label.config(text=item["message"])
                self.update_log(item["message"], 'success' if item["success"] else 'error')
            elif item["type"] == "chunk_result":
                # Single chunk test result
                self.progress_bar.stop()
                self.progress_bar.config(mode="determinate", value=0)
                self.test_chunk_button.config(state='normal')
                self.chunk_result_text.config(state='normal')
                self.chunk_result_text.delete(1.0, tk.END)
                self.chunk_result_text.insert(tk.END, item["message"])
                self.chunk_result_text.config(state='disabled')
                if item["success"]:
                    tag = 'success' if item.get("passed", False) else 'warning'
                    self.status_label.config(text="Test complete.")
                    self.update_log(f"Chunk test complete: {'PASSED' if item.get('passed') else 'FAILED'}", tag)
                else:
                    self.status_label.config(text="Test failed.")
                    self.update_log(item["message"], 'error')

        self.root.after(100, self.process_queue)  # Check queue every 100ms

    def start_validation_thread(self):
        """Starts a thread to validate TTS files in the selected folder.
        Args:
        None
        Returns:
        None
        """
        if not self.tts_folder_path or not self.tts_folder_path.is_dir():
            messagebox.showerror("Error", "Please select a valid TTS folder.")
            return

        # Check for expected subdirectories
        audio_chunks_dir = self.tts_folder_path / "audio_chunks"
        text_chunks_dir = self.tts_folder_path / "text_chunks"
        if not audio_chunks_dir.is_dir() or not text_chunks_dir.is_dir():
            messagebox.showerror("Error", "Selected folder must contain 'audio_chunks' and 'text_chunks' subdirectories.")
            return

        self.run_button.config(state='disabled')
        self.browse_button.config(state='disabled')
        self.status_label.config(text="Starting validation...")
        self.progress_bar.config(value=0)
        self.progress_bar.start()

        validation_thread = threading.Thread(target=self._run_validation_logic)
        validation_thread.daemon = True  # Allow the thread to exit with the main program
        validation_thread.start()

    def _run_validation_logic(self):
        """Runs batch validation logic for TTS files.
        Args:
        None
        Returns:
        None
        """
        try:
            if not self.tts_folder_path:
                self.progress_queue.put({"type": "finished", "success": False, "message": "No TTS folder selected"})
                return

            tts_folder = self.tts_folder_path
            threshold = self.threshold_var.get()

            self.progress_queue.put({"type": "status", "message": "Starting batch validation..."})
            model_name = self.model_var.get()
            validation_results = validate_batch(tts_folder, threshold, self.progress_queue, model_name=model_name)

            if validation_results.get("error"):
                self.progress_queue.put({"type": "finished", "success": False, "message": f"Validation Failed: {validation_results['error']}"})
                return

            self.progress_queue.put({"type": "status", "message": "Generating reports..."})
            self.generate_validation_log(tts_folder, validation_results["results"], threshold)
            self.generate_fail_log(tts_folder, validation_results["failed_chunks"], threshold)

            final_message = (f"Validation complete! "
                             f"Total: {validation_results['total']}, "
                             f"Passed: {validation_results['passed']}, "
                             f"Failed: {validation_results['failed']}. "
                             f"Logs saved in {tts_folder}")
            self.progress_queue.put({"type": "finished", "success": True, "message": final_message})

        except Exception as e:
            logging.exception("An unexpected error occurred during validation.")
            self.progress_queue.put({"type": "finished", "success": False, "message": f"An unexpected error occurred: {e}"})

    def generate_validation_log(self, tts_dir: Path, results: List[Dict[str, Any]], threshold: float):
        """Writes a validation log for ASR results.
        Args:
        tts_dir (Path): The directory containing the TTS output.
        results (List[Dict[str, Any]]): A list of dictionaries containing validation results.
        threshold (float): The similarity threshold used for validation.
        Returns:
        None
        """
        log_path = tts_dir / "validation.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"ASR Validation Report for: {tts_dir}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Similarity Threshold: {threshold:.2f}\n")
            f.write("=" * 80 + "\n\n")
            for result in results:
                f.write(f"Chunk: {result['chunk_num']}\n")
                f.write(f"Status: {'PASSED' if result['passed'] else 'FAILED'} (Score: {result['score']:.2f})\n")
                f.write(f"Classification: {result.get('classification', 'FAIL')}\n")
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                if result.get('hallucination_warning'):
                    f.write(f"Hallucination: {result['hallucination_warning']}\n")
                if result.get('truncation_warning'):
                    f.write(f"Truncation: {result['truncation_warning']}\n")
                f.write(f"Original Text: {result.get('ref_text_raw', result.get('ref_normalized', 'N/A'))}\n")
                f.write(f"Transcribed Text: {result.get('hyp_text_raw', result.get('hyp_normalized', 'N/A'))}\n")
                identifier_comparisons = result.get("identifier_comparisons") or []
                counted = [item for item in identifier_comparisons if item.get("result") != "exempt"]
                matched = sum(1 for item in counted if item.get("matched"))
                f.write(
                    f"Identifier Score: {float(result.get('id_score') or 0):.2f} "
                    f"({matched}/{len(counted)} matched, {len(identifier_comparisons)} slots)\n"
                )
                accepted_lists = result.get("accepted_list_label_equivalences") or []
                if accepted_lists:
                    f.write(f"Accepted List Labels: {len(accepted_lists)} accepted positions\n")
                if result.get("requires_second_stage_confirmation"):
                    reason = result.get("second_stage_confirmation_reason") or "yes"
                    f.write(f"Second-Stage Confirmation: {reason}\n")
                f.write("-" * 40 + "\n\n")
        self.progress_queue.put({"type": "status", "message": f"Generated validation.log at {log_path}"})

    def generate_fail_log(self, tts_dir: Path, failed_results: List[Dict[str, Any]], threshold: float):
        """Generate a fail log for ASR failed chunks.
        Args:
        tts_dir (Path): The directory where the TTS files are stored.
        failed_results (List[Dict[str, Any]]): A list of dictionaries containing the results of failed validation.
        threshold (float): The similarity threshold used for validation.
        Returns: None
        """
        log_path = tts_dir / "fail.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"ASR Failed Chunks Report for: {tts_dir}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Similarity Threshold: {threshold:.2f}\n")
            f.write("=" * 80 + "\n\n")
            if not failed_results:
                f.write("No chunks failed validation below the set threshold.\n")
            for result in failed_results:
                f.write(f"Chunk: {result['chunk_num']}\n")
                f.write(f"Status: FAILED (Score: {result['score']:.2f})\n")
                f.write(f"Classification: {result.get('classification', 'FAIL')}\n")
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                if result.get('hallucination_warning'):
                    f.write(f"Hallucination: {result['hallucination_warning']}\n")
                if result.get('truncation_warning'):
                    f.write(f"Truncation: {result['truncation_warning']}\n")
                f.write(f"Original Text: {result.get('ref_text_raw', result.get('ref_normalized', 'N/A'))}\n")
                f.write(f"Transcribed Text: {result.get('hyp_text_raw', result.get('hyp_normalized', 'N/A'))}\n")
                identifier_comparisons = result.get("identifier_comparisons") or []
                counted = [item for item in identifier_comparisons if item.get("result") != "exempt"]
                matched = sum(1 for item in counted if item.get("matched"))
                f.write(
                    f"Identifier Score: {float(result.get('id_score') or 0):.2f} "
                    f"({matched}/{len(counted)} matched, {len(identifier_comparisons)} slots)\n"
                )
                accepted_lists = result.get("accepted_list_label_equivalences") or []
                if accepted_lists:
                    f.write(f"Accepted List Labels: {len(accepted_lists)} accepted positions\n")
                if result.get("requires_second_stage_confirmation"):
                    reason = result.get("second_stage_confirmation_reason") or "yes"
                    f.write(f"Second-Stage Confirmation: {reason}\n")
                f.write(f"Explanation: {explain_diff(result['ref_normalized'], result['hyp_normalized'])}\n")
                f.write("-" * 40 + "\n\n")
        self.progress_queue.put({"type": "status", "message": f"Generated fail.log at {log_path}"})


def _failure_entry_from_result(wav_file: Path, chunk_num: str, result: dict) -> dict:
    """Build one Stage 1 candidate row from an ASR comparison result."""
    return {
        "chunk_index": (
            int(chunk_num.split("_")[1]) if "_" in chunk_num else int(chunk_num)
        ),
        "filename": str(wav_file),
        "score": result.get("score", 0.0),
        "original_temp": 0.7,
        "error": result.get("error", ""),
        "explanation": result.get("explanation", ""),
        "original_text": result.get("ref_text_raw", ""),
        "transcribed_text": result.get("hyp_text_raw", ""),
        "ref_normalized": result.get("ref_normalized", ""),
        "hyp_normalized": result.get("hyp_normalized", ""),
        "stage_one_backend": result.get("stage_one_backend", ""),
        "stage_one_model": result.get("stage_one_model", ""),
        "stage_one_elapsed_s": result.get("stage_one_elapsed_s", 0.0),
        "prose_score": result.get("prose_score", 0.0),
        "id_score": result.get("id_score", 0.0),
        "identifier_comparisons": result.get("identifier_comparisons", []),
        "accepted_list_label_equivalences": result.get("accepted_list_label_equivalences", []),
        "accepted_book_term_equivalences": result.get("accepted_book_term_equivalences", []),
        "requires_second_stage_confirmation": result.get("requires_second_stage_confirmation", False),
        "second_stage_confirmation_reason": result.get("second_stage_confirmation_reason", ""),
        "classification": result.get("classification", "FAIL"),
        "minor_mismatch": result.get("minor_mismatch", False),
        "hallucination_warning": result.get("hallucination_warning", ""),
        "truncation_warning": result.get("truncation_warning", ""),
    }


def run_parakeet_batch_validation(
    tts_dir: Path | str,
    chunks: List[str],
    threshold: float,
    model_name: str,
) -> List[Dict[str, Any]]:
    """Run resident Parakeet Stage 1 transcription once for every supplied chunk.

    Unlike the packed Whisper compatibility pipeline, each Parakeet candidate is
    sent directly to Stage 2 after its first comparison; it receives no solo
    transcription retry that would delay the book without proving bad speech.
    Recurring source-term evidence is built once for the book so harmless ASR
    spellings of established terms do not inflate the Stage 2 candidate set.
    """
    try:
        from .verification_backends import ParakeetTDTBackend
    except ImportError:
        from verification_backends import ParakeetTDTBackend

    root = Path(tts_dir)
    paths = [root / "audio_chunks" / f"{chunk}.wav" for chunk in chunks]
    book_term_evidence = build_tts_book_term_evidence(root)
    backend_model = model_name if model_name.startswith("nvidia/") else None
    backend = ParakeetTDTBackend(**({"model_name": backend_model} if backend_model else {}))
    evidence_by_path: Dict[str, Any] = {}
    try:
        evidence_by_path = {
            item.audio_path: item for item in backend.transcribe_paths(paths)
        }
        results: List[Dict[str, Any]] = []
        for chunk, path in zip(chunks, paths):
            text_path = root / "text_chunks" / f"{chunk}.txt"
            if not text_path.exists():
                result = _empty_result(chunk, str(path))
                result["error"] = "Text file does not exist"
                results.append(result)
                continue
            item = evidence_by_path.get(str(path.resolve()))
            if item is None:
                result = _empty_result(chunk, str(path))
                result["error"] = "Parakeet returned no transcript"
                results.append(result)
                continue
            result = score_asr_pair(
                chunk, text_path.read_text(encoding="utf-8").strip(), item.text,
                threshold,
                audio_path=str(path),
                book_term_evidence=book_term_evidence,
            )
            result["stage_one_backend"] = item.backend
            result["stage_one_model"] = item.model
            result["stage_one_elapsed_s"] = item.elapsed_s
            result["stage_one_timestamps"] = item.timestamps
            result["stage_one_confidence"] = item.confidence
            results.append(result)
        return results
    finally:
        backend.close()


def _append_failure_locked(log_path: Path, failure_entry: dict, lock: threading.Lock) -> None:
    """Append one failure entry under a lock (single-writer safety)."""
    with lock:
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    failures = json.load(f)
            except Exception:
                failures = []
        else:
            failures = []
        failures.append(failure_entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)


def _asr_cpu_worker_loop(
    worker_id: int,
    job_queue: "queue.Queue",
    result_queue: "queue.Queue",
    model_name: str,
    language: str,
    threshold: float,
    force_device: str,
    engine: str = "faster_whisper",
    n_threads: int = 2,
) -> None:
    """Load one ASR model and validate jobs until sentinel None.

    Args:
        worker_id: Worker index for logs.
        job_queue: Queue of (wav_path_str, chunk_num, tts_dir_str) or None.
        result_queue: Queue of (wav_path_str, result_dict, worker_id).
        model_name: Model size name.
        language: Whisper language pin.
        threshold: Pass threshold.
        force_device: 'cpu' or 'cuda' for this worker.
        engine: ASR engine (faster_whisper only).
        n_threads: Unused with faster-whisper; kept for call-site compat.
    """
    asr_model, device = load_asr_model_adaptive(
        model_name,
        force_device=force_device,
        engine=engine,
        n_threads=n_threads,
    )
    if not asr_model:
        result_queue.put(("__worker_fatal__", {"error": "model_load_failed", "worker_id": worker_id}, worker_id))
        return
    print(
        f"✅ ASR worker {worker_id} ready on {_format_device(device)} "
        f"engine={engine} model={model_name}"
    )
    try:
        while True:
            job = job_queue.get()
            if job is None:
                break
            wav_path_str, chunk_num, tts_dir_str = job
            try:
                result = validate_single_chunk(
                    chunk_num,
                    Path(tts_dir_str),
                    threshold,
                    asr_model,
                    language=language,
                )
            except Exception as exc:
                result = {
                    "score": 0.0,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result_queue.put((wav_path_str, result, worker_id))
    finally:
        cleanup_asr_model(asr_model)
        print(f"🛑 ASR worker {worker_id} stopped")


def monitor_and_validate_folder(
    folder_path: str,
    log_file: str,
    threshold: float,
    expected_count: int = 0,
    model_name: str = DEFAULT_ASR_MODEL,
    language: str = "en",
    force_device: Optional[str] = None,
    cpu_workers: int = 1,
    engine: str = "faster_whisper",
    n_threads: int = 0,
    gpu_workers_after_tts: int = 1,
):
    """Validate new WAV files until expected_count files have been processed.

    Args:
        folder_path: audio_chunks directory to watch.
        log_file: Path for asr_failures.json.
        threshold: Similarity pass threshold.
        expected_count: Stop after this many WAVs processed (0 = run until signal).
        model_name: Model size (base, tiny, …).
        language: Whisper language code to pin during decode.
        force_device: Optional 'cpu'/'cuda' override (prefer cpu while TTS uses GPU).
        cpu_workers: Number of parallel ASR workers (>=1); each loads own model.
        engine: ASR engine (faster_whisper only).
        n_threads: Unused with faster-whisper; kept for call-site compat.
        gpu_workers_after_tts: After TTS writes tts_gen_done.flag, start this many
            faster-whisper CUDA workers to drain the remaining backlog (0=off).
    """
    import os
    import signal

    folder = Path(folder_path)
    log_path = Path(log_file)
    tts_dir = folder.parent
    gen_done_flag = tts_dir / "tts_gen_done.flag"
    processed_files = set()
    n_workers = max(1, int(cpu_workers))
    # Cap at 8: calibrated per-GPU; mid cards ~4, high-end may use 6–8.
    gpu_after = max(0, min(16, int(gpu_workers_after_tts)))
    # Prefer forced CPU for concurrent TTS; default adaptive if unset.
    device_pref = (force_device or "").strip().lower() or None
    eng = (engine or "faster_whisper").strip().lower().replace("-", "_")
    # Auto threads: leave headroom when many workers share one CPU.
    if n_threads and int(n_threads) > 0:
        threads_per = max(1, int(n_threads))
    else:
        cores = os.cpu_count() or 4
        threads_per = max(1, cores // max(1, n_workers))

    # Load existing log to avoid reprocessing
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                failures = json.load(f)
                for failure in failures:
                    processed_files.add(failure.get("filename", ""))
        except Exception as e:
            print(f"Warning: Could not load existing failure log: {e}")

    print(f"🔍 Starting ASR monitoring of: {folder_path}")
    print(f"📝 Logging failures to: {log_file}")
    print(f"🎯 Threshold: {threshold}")
    print(f"🌐 Language: {language}")
    print(f"👥 ASR CPU workers: {n_workers}")
    print(f"📌 Device policy: {device_pref or 'auto'}")
    print(f"⚙️  Engine: {eng}  threads/worker: {threads_per}")
    print(f"🚀 GPU workers after TTS done: {gpu_after} (flag: {gen_done_flag.name})")
    print(f"📊 Tracking {len(processed_files)} previously processed files")

    shutdown_requested = False

    def signal_handler(sig, frame):
        """Request graceful monitor shutdown on SIGINT/SIGTERM."""
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\n🛑 Shutdown signal received...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    t0 = time.time()
    if n_workers == 1:
        # ---- Single-worker path (original behavior + device force) ----
        asr_model, device = load_asr_model_adaptive(
            model_name,
            force_device=device_pref,
            engine=eng,
            n_threads=threads_per,
        )
        if not asr_model:
            print("❌ Failed to load ASR model")
            return
        print(f"✅ ASR model loaded on {_format_device(device)} engine={eng}")
        single_log_lock = threading.Lock()
        try:
            while not shutdown_requested:
                try:
                    wav_files = sorted(folder.glob("chunk_*.wav"))
                    for wav_file in wav_files:
                        if str(wav_file) in processed_files:
                            continue
                        print(f"🎵 Processing new file: {wav_file.name}")
                        chunk_num = wav_file.stem
                        tts_dir = folder.parent
                        result = validate_single_chunk(
                            chunk_num, tts_dir, threshold, asr_model, language=language
                        )
                        score = result.get("score", 0.0)
                        passed = result.get("passed", False)
                        if passed:
                            print(f"✅ {wav_file.name}: PASSED (score: {score:.2f})")
                        else:
                            print(f"❌ {wav_file.name}: FAILED (score: {score:.2f})")
                            try:
                                _append_failure_locked(
                                    log_path,
                                    _failure_entry_from_result(wav_file, chunk_num, result),
                                    single_log_lock,
                                )
                                print(f"📝 Logged failure: {wav_file.name}")
                            except Exception as e:
                                print(f"⚠️ Failed to log failure: {e}")
                        processed_files.add(str(wav_file))
                        if expected_count and len(processed_files) >= expected_count:
                            print(
                                f"✅ ASR completed: processed {len(processed_files)} chunks "
                                f"in {time.time() - t0:.1f}s"
                            )
                            shutdown_requested = True
                            break
                except Exception as e:
                    print(f"⚠️ Error during file processing: {e}")
                time.sleep(1.0)
        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
        finally:
            cleanup_asr_model(asr_model)
            print("🛑 Monitoring stopped")
        return

    # ---- Multi-worker path: claim queue + N models ----
    # Phase 1: CPU workers during TTS (no GPU fight).
    # Phase 2: when tts_gen_done.flag appears, add faster-whisper CUDA workers
    #          to drain remaining backlog (CPU workers keep running = mix).
    worker_device = device_pref or "cpu"
    job_q: queue.Queue = queue.Queue(maxsize=max(16, n_workers * 8))
    result_q: queue.Queue = queue.Queue()
    log_lock = threading.Lock()
    in_flight: set = set()
    workers = []
    gpu_boost_started = False
    total_worker_slots = n_workers  # grows if GPU boost starts

    def _start_worker(
        wid: int,
        device: str,
        worker_engine: str,
        threads: int,
        label: str,
    ) -> None:
        """Start one ASR worker thread and append to workers list."""
        t = threading.Thread(
            target=_asr_cpu_worker_loop,
            args=(
                wid,
                job_q,
                result_q,
                model_name,
                language,
                threshold,
                device,
                worker_engine,
                threads,
            ),
            name=f"asr-worker-{label}-{wid}",
            daemon=True,
        )
        t.start()
        workers.append(t)
        print(f"▶️  Started ASR worker {wid} ({label}) device={device} engine={worker_engine}")

    for wid in range(n_workers):
        _start_worker(wid, worker_device, eng, threads_per, "cpu")

    try:
        while not shutdown_requested:
            try:
                # --- Phase 2 trigger: TTS original gen finished ---
                if (
                    not gpu_boost_started
                    and gpu_after > 0
                    and gen_done_flag.exists()
                ):
                    gpu_boost_started = True
                    print(
                        f"🚀 TTS gen done flag seen — starting {gpu_after} GPU "
                        f"faster_whisper worker(s) for backlog drain"
                    )
                    # Prefer faster_whisper on CUDA for the finish phase (cpp GPU
                    # often unavailable; base fits in leftover VRAM after TTS).
                    for i in range(gpu_after):
                        gwid = 100 + i
                        _start_worker(
                            gwid,
                            "cuda",
                            "faster_whisper",
                            max(2, threads_per),
                            "gpu",
                        )
                        total_worker_slots += 1

                # Drain completed validations first
                while True:
                    try:
                        wav_path_str, result, wid = result_q.get_nowait()
                    except queue.Empty:
                        break
                    if wav_path_str == "__worker_fatal__":
                        print(f"❌ ASR worker fatal: {result}")
                        # GPU load may fail if VRAM tight — continue with CPU only
                        continue
                    wav_file = Path(wav_path_str)
                    chunk_num = wav_file.stem
                    score = result.get("score", 0.0)
                    passed = result.get("passed", False)
                    if passed:
                        print(f"✅ {wav_file.name}: PASSED (score: {score:.2f}) [w{wid}]")
                    else:
                        print(f"❌ {wav_file.name}: FAILED (score: {score:.2f}) [w{wid}]")
                        try:
                            _append_failure_locked(
                                log_path,
                                _failure_entry_from_result(wav_file, chunk_num, result),
                                log_lock,
                            )
                            print(f"📝 Logged failure: {wav_file.name}")
                        except Exception as e:
                            print(f"⚠️ Failed to log failure: {e}")
                    processed_files.add(wav_path_str)
                    in_flight.discard(wav_path_str)
                    if expected_count and len(processed_files) >= expected_count:
                        print(
                            f"✅ ASR completed: processed {len(processed_files)} chunks "
                            f"in {time.time() - t0:.1f}s "
                            f"(cpu_workers={n_workers}, gpu_boost={gpu_boost_started})"
                        )
                        shutdown_requested = True
                        break

                if shutdown_requested:
                    break

                # Claim newly arrived WAVs (exactly-once: processed ∪ in_flight)
                wav_files = sorted(folder.glob("chunk_*.wav"))
                for wav_file in wav_files:
                    key = str(wav_file)
                    if key in processed_files or key in in_flight:
                        continue
                    # Text sidecar ready (TTS publishes txt before wav)
                    text_sidecar = tts_dir / "text_chunks" / f"{wav_file.stem}.txt"
                    if not text_sidecar.exists():
                        continue
                    try:
                        job_q.put_nowait((key, wav_file.stem, str(tts_dir)))
                        in_flight.add(key)
                        print(f"🎵 Claimed {wav_file.name} for ASR")
                    except queue.Full:
                        break

            except Exception as e:
                print(f"⚠️ Error during multi-worker dispatch: {e}")

            time.sleep(0.25)

    except Exception as e:
        print(f"⚠️ Monitoring error: {e}")
    finally:
        for _ in workers:
            job_q.put(None)
        for t in workers:
            t.join(timeout=120)
        try:
            if gen_done_flag.exists():
                gen_done_flag.unlink()
        except Exception:
            pass
        print("🛑 Multi-worker monitoring stopped")


def run_batch_folder_validation(
    tts_dir: Path | str,
    log_file: Path | str,
    threshold: float,
    model_name: str = DEFAULT_ASR_MODEL,
    language: str = "en",
    force_device: Optional[str] = "cuda",
    engine: str = "faster_whisper",
    cpu_workers: int = 8,
    n_threads: int = 0,
    pack_size: int = 8,
    pack_silence_s: float = 1.0,
) -> Dict[str, Any]:
    """Validate all chunks in a finished TTS folder; write asr_failures.json.

    Post-gen path for Pocket TTS GUI: same scoring as standalone.
    - faster_whisper + GPU: Phase 1–3 pipeline (1 model, pack + solo-retry).
    - whisper_cpp / CPU: multi-model ThreadPool (one model per worker).

    Args:
        tts_dir: Directory with audio_chunks/ and text_chunks/.
        log_file: Path for asr_failures.json (fail list for regen).
        threshold: Pass threshold.
        model_name: Model size name.
        language: Language pin.
        force_device: cuda/cpu/auto.
        engine: faster_whisper or whisper_cpp.
        cpu_workers: Pipeline CPU budget or multi-model count for cpp.
        n_threads: whisper.cpp threads per instance.
        pack_size: Clips per GPU pack (faster_whisper only).
        pack_silence_s: Silence between packed clips.

    Returns:
        Summary dict with total/passed/failed/wall_s.
    """
    tts_dir = Path(tts_dir)
    log_path = Path(log_file)
    eng = (engine or "faster_whisper").strip().lower().replace("-", "_")
    if eng in ("whispercpp", "cpp", "pywhispercpp"):
        eng = "whisper_cpp"
    force = (force_device or "cuda").strip().lower()
    if force == "gpu":
        force = "cuda"
    workers = max(1, min(16, int(cpu_workers)))
    chunks = discover_chunks(tts_dir)
    total = len(chunks)
    print(f"📦 Batch folder validation: {total} chunks engine={eng} device={force}")
    print(f"📝 Failures → {log_path}")
    if total == 0:
        log_path.write_text("[]\n", encoding="utf-8")
        return {"total": 0, "passed": 0, "failed": 0, "wall_s": 0.0}

    # Clear prior fail log so regen sees only this run
    try:
        log_path.write_text("[]\n", encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    use_pipeline = eng == "faster_whisper" and force in ("cuda", "auto", "")

    if eng in ("parakeet", "parakeet_tdt", "nemo_parakeet"):
        print("🚀 Stage 1: resident Parakeet path batches; no solo candidate retries")
        results = run_parakeet_batch_validation(
            tts_dir, chunks, threshold, model_name=model_name,
        )
    elif use_pipeline:
        load_n = max(2, workers // 2)
        score_n = max(2, workers - load_n)
        print(
            f"🚀 Pipeline: 1× GPU model pack={pack_size} loaders={load_n} "
            f"scorers={score_n} vad_filter=False solo_retry_on_fail"
        )
        asr_model, device = load_asr_model_adaptive(
            model_name,
            force_device="cuda" if force in ("cuda", "auto", "") else force,
            engine=eng,
            n_threads=max(1, int(n_threads) or 2),
        )
        if not asr_model:
            print("❌ Failed to load ASR model for pipeline")
            return {"total": total, "passed": 0, "failed": total, "wall_s": 0.0, "error": "model_load"}

        def _prog(done: int, tot: int, result: Dict[str, Any]) -> None:
            """Print periodic progress for generator log capture."""
            if done == tot or done % 50 == 0 or done <= 3:
                status = "PASS" if result.get("passed") else "FAIL"
                print(
                    f"{status} {result.get('chunk_num')} "
                    f"score={float(result.get('score') or 0):.3f} "
                    f"({done}/{tot})",
                    flush=True,
                )

        try:
            results = run_pipeline_batch_validation(
                tts_dir,
                chunks,
                asr_model,
                threshold,
                language=language or "en",
                load_workers=load_n,
                score_workers=score_n,
                vad_filter=False,
                pack_size=max(1, int(pack_size)),
                pack_silence_s=float(pack_silence_s),
                progress_cb=_prog,
            )
        finally:
            cleanup_asr_model(asr_model)
    else:
        # Multi-copy path (whisper_cpp GPU or any CPU multi-worker)
        import os

        cores = os.cpu_count() or 4
        threads_per = (
            max(1, int(n_threads))
            if n_threads and int(n_threads) > 0
            else max(1, cores // max(1, workers))
        )
        print(f"👥 Multi-model: {workers}× {eng} on {force} threads/worker={threads_per}")
        models: List[Any] = []
        free: queue.Queue = queue.Queue()
        try:
            for i in range(workers):
                m, dev = load_asr_model_adaptive(
                    model_name,
                    force_device=force if force != "auto" else None,
                    engine=eng,
                    n_threads=threads_per,
                )
                if m is None:
                    print(f"❌ Failed to load worker model {i}")
                    break
                models.append(m)
                free.put(m)
                print(f"✅ Worker {i} on {dev}")
            if not models:
                return {"total": total, "passed": 0, "failed": total, "wall_s": 0.0, "error": "model_load"}

            def work(chunk_num: str) -> Dict[str, Any]:
                """Validate one chunk with a borrowed model."""
                m = free.get()
                try:
                    return validate_single_chunk(
                        chunk_num,
                        tts_dir,
                        threshold,
                        m,
                        language=language or "en",
                        vad_filter=False,
                    )
                finally:
                    free.put(m)

            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                futures = {pool.submit(work, c): c for c in chunks}
                done = 0
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "chunk_num": futures[fut],
                            "passed": False,
                            "score": 0.0,
                            "error": str(exc),
                            "classification": "FAIL",
                        }
                    results.append(result)
                    done += 1
                    if done == total or done % 50 == 0 or done <= 3:
                        status = "PASS" if result.get("passed") else "FAIL"
                        print(
                            f"{status} {result.get('chunk_num')} "
                            f"score={float(result.get('score') or 0):.3f} "
                            f"({done}/{total})",
                            flush=True,
                        )
        finally:
            for m in models:
                try:
                    cleanup_asr_model(m)
                except Exception:
                    pass

    # Write failures for generator regen (same schema as monitor)
    failures: List[dict] = []
    audio_dir = tts_dir / "audio_chunks"
    for result in results:
        if result.get("passed"):
            continue
        chunk_num = result.get("chunk_num") or ""
        wav = audio_dir / f"{chunk_num}.wav"
        failures.append(_failure_entry_from_result(wav, chunk_num, result))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")

    wall = time.time() - t0
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    print(
        f"✅ Batch done: pass={passed} fail={failed} wall={wall:.1f}s "
        f"({(len(results) / wall) if wall > 0 else 0:.2f} chunks/s)",
        flush=True,
    )
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "wall_s": wall,
        "pipeline": use_pipeline or eng in ("parakeet", "parakeet_tdt", "nemo_parakeet"),
        "stage_one_backend": eng,
        "log_file": str(log_path),
    }


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Validator - TTS Quality Control")
    parser.add_argument('--monitor-folder', help='Folder to monitor for new WAV files (audio_chunks subdirectory)')
    parser.add_argument(
        '--batch-tts-dir',
        help='Run full-folder batch (post-gen pipeline) on this TTS directory',
    )
    parser.add_argument('--tts-dir', help='Parent TTS directory (contains audio_chunks and text_chunks)')
    parser.add_argument('--log-file', default='asr_failures.json', help='Log file for failures (default: asr_failures.json)')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (default: 0.8)')
    parser.add_argument('--model', default=DEFAULT_ASR_MODEL, help='faster-whisper model name')
    parser.add_argument('--language', default='en',
                        help='Whisper language code to pin (default: en)')
    parser.add_argument('--single-chunk', help='Validate a single chunk (e.g., chunk_00005)')
    parser.add_argument('--audio-path', help='Override audio file path for single-chunk validation')
    parser.add_argument('--json', action='store_true', help='Emit JSON output for single-chunk mode')
    parser.add_argument('--expected-count', type=int, default=0,
                        help='Stop monitor after this many WAV files are processed')
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'gpu'],
        help='ASR device: auto (VRAM adaptive), cpu (recommended during TTS), cuda/gpu',
    )
    parser.add_argument(
        '--cpu-workers',
        type=int,
        default=1,
        help='Parallel ASR workers each with own Whisper (use CPU; default 1)',
    )
    parser.add_argument(
        '--engine',
        default='faster_whisper',
        choices=['faster_whisper', 'whisper_cpp', 'whisper-cpp', 'cpp', 'parakeet', 'parakeet_tdt'],
        help='ASR engine: faster_whisper, whisper_cpp, or optional NeMo Parakeet',
    )
    parser.add_argument(
        '--n-threads',
        type=int,
        default=0,
        help='CPU threads per whisper.cpp worker (0=auto cores/workers)',
    )
    parser.add_argument(
        '--gpu-workers-after-tts',
        type=int,
        default=1,
        help='After tts_gen_done.flag, start N faster-whisper CUDA workers (0=off, max 8)',
    )
    parser.add_argument(
        '--pack-size',
        type=int,
        default=8,
        help='Clips per GPU pack for faster_whisper batch pipeline (default 8)',
    )

    args = parser.parse_args()

    if args.batch_tts_dir:
        force = None if args.device == "auto" else args.device
        summary = run_batch_folder_validation(
            args.batch_tts_dir,
            args.log_file,
            args.threshold,
            model_name=args.model,
            language=args.language,
            force_device=force or "cuda",
            engine=args.engine,
            cpu_workers=max(1, int(args.cpu_workers)),
            n_threads=int(args.n_threads),
            pack_size=max(1, int(args.pack_size)),
        )
        if args.json:
            print(json.dumps(summary))
        sys.exit(0 if summary.get("failed", 0) == 0 else 1)

    if args.monitor_folder:
        tts_dir = args.tts_dir if args.tts_dir else str(Path(args.monitor_folder).parent)
        force = None if args.device == "auto" else args.device
        monitor_and_validate_folder(
            args.monitor_folder,
            args.log_file,
            args.threshold,
            args.expected_count,
            args.model,
            language=args.language,
            force_device=force,
            cpu_workers=max(1, int(args.cpu_workers)),
            engine=args.engine,
            n_threads=int(args.n_threads),
            gpu_workers_after_tts=max(0, int(args.gpu_workers_after_tts)),
        )
    elif args.single_chunk:
        if not args.tts_dir:
            message = "--tts-dir required for single-chunk mode"
            if args.json:
                print(json.dumps({"error": message}))
            else:
                print(f"Error: {message}")
            sys.exit(1)

        audio_override = Path(args.audio_path) if args.audio_path else None

        asr_model, device = load_asr_model_adaptive(
            args.model,
            force_device=None if args.device == "auto" else args.device,
            engine=args.engine,
            n_threads=int(args.n_threads) or 2,
        )
        if not asr_model:
            message = "Failed to load ASR model"
            if args.json:
                print(json.dumps({"score": 0.0, "passed": False, "error": message}))
            else:
                print(f"❌ {message}")
            sys.exit(1)

        print(f"✅ ASR model loaded on {_format_device(device)} engine={args.engine}")

        result = validate_single_chunk(
            args.single_chunk,
            Path(args.tts_dir),
            args.threshold,
            asr_model,
            audio_path_override=audio_override,
            language=args.language,
        )

        score = result.get('score', 0.0)
        passed = result.get('passed', False)

        if args.json:
            print(json.dumps(result))
        else:
            print(f"\n{'='*50}")
            print(f"Chunk: {args.single_chunk}")
            print(f"Score: {score:.4f}")
            print(f"Status: {'PASSED' if passed else 'FAILED'}")
            print(f"{'='*50}")

        cleanup_asr_model(asr_model)
        sys.exit(0 if passed else 1)
    else:
        root = tk.Tk()
        # Hide hidden files/folders in file dialogs by default (same as asr_gui)
        try:
            root.tk.eval("catch {tk_getOpenFile -badoption}")
            root.tk.setvar("::tk::dialog::file::showHiddenBtn", 1)
            root.tk.setvar("::tk::dialog::file::showHiddenVar", 0)
        except Exception:
            pass
        app = ASRApp(root)
        root.mainloop()


# ============================================================================
# PHASES 4-7: DETECTION FUNCTIONS (SPECIFICATION)
# ============================================================================

"""
PHASE 4: detect_extra_speech(alignment)
Input: alignment dict with ref/hyp token lists + match flags
Returns: {
  "has_extra": bool,
  "locations": List[{"type": "prefix|internal|suffix", "tokens": [...], "span": (i,j)}],
  "failure_reasons": List[str]
}
Rules:
- Prefix: leading unmatched hyp tokens before first ref match
- Internal: >=2 consecutive unmatched content words (non-function words)
- Suffix: trailing unmatched hyp tokens after last ref match, length > tolerance (default 2 tokens or 15% of ref length)
- Always fail on unmatched clause (punctuation-bounded segment with no ref overlap)
"""

"""
PHASE 5: detect_missing_speech(alignment)
Input: alignment dict
Returns: {
  "has_missing": bool,
  "locations": List[{"type": "...", "tokens": [...], "severity": "high|medium"}],
  "penalty_score": float (0.0-1.0 additive)
}
Strong penalties (immediate fail triggers):
- Missing negation (not, never, no, n't, cannot)
- Missing number or numeric entity
- Missing proper name (capitalized tokens)
- Missing content word (noun/verb/adjective) in clause-final position
- Missing ending (last 3 tokens of ref have <50% coverage)
- >=2 consecutive missing content words
Function words (the, a, to, of) missing alone do not trigger fail.
"""

"""
PHASE 6: detect_repetition(ref_tokens, hyp_tokens)
Input: two token lists
Returns: {
  "has_repetition": bool,
  "type": "block|expected_ref_repeat",
  "details": {"n": int, "span": (i,j), "tokens": [...]}
}
Detection:
- Block equality: any n-gram (n=2..6) appears >=2 times consecutively in hyp but not in ref
- Expected-ref-repeat allowance: if ref has "Linx Linx" (case-insensitive content match) then hyp "links links" is allowed; same for onomatopoeia patterns (tra-lal vs la-la-la, tra-la-la)
- Ignore case/punctuation for comparison; require exact token count match for allowance
"""

"""
PHASE 7: check_critical_tokens(alignment)
Input: alignment dict
Returns: {
  "critical_fail": bool,
  "overrides": List[{"token": str, "ref": "...", "hyp": "...", "reason": "negation|number|name|polarity"}]
}
Fail overrides (highest priority, force FAIL even if score high):
- Changed negation (ref "not" vs hyp "now", ref "never" vs hyp "ever")
- Changed number (ref "two" vs hyp "three", "2023" vs "2024")
- Changed name (ref "Alice" vs hyp "Alex")
- Polarity inversion (ref "is happy" vs hyp "isn't happy", ref "can" vs hyp "can't")
Token alignment must show substitution or deletion at these positions.
"""
