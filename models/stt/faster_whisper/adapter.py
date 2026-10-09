"""Faster-Whisper Model Integration Adapter.

Provides a clean interface for loading and transcribing using the
local CTranslate2 Faster-Whisper model weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_REGISTRY = {
    "turbo": REPO_ROOT / "weights" / "stt" / "faster-whisper-large-v3-turbo-ct2",
    "large-v3-turbo": REPO_ROOT / "weights" / "stt" / "faster-whisper-large-v3-turbo-ct2",
    "tiny": REPO_ROOT / "weights" / "stt" / "faster-whisper-tiny",
}
DEFAULT_WEIGHTS_DIR = MODEL_REGISTRY["tiny"]


class FasterWhisperAdapter:
    """Adapter for Faster-Whisper ASR inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        compute_type: str = "float16",
        cpu_threads: int = 4,
    ):
        if model_path is None:
            self.model_path = DEFAULT_WEIGHTS_DIR
        elif str(model_path).lower() in MODEL_REGISTRY:
            self.model_path = MODEL_REGISTRY[str(model_path).lower()]
        else:
            self.model_path = Path(model_path) if Path(str(model_path)).exists() else model_path

        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model: Optional[WhisperModel] = None

    @classmethod
    def get_default_weights_path(cls, model_name: str = "tiny") -> Path:
        """Return the default repository weights path for a given model."""
        return MODEL_REGISTRY.get(model_name.lower(), DEFAULT_WEIGHTS_DIR)

    def load(self) -> WhisperModel:
        """Load and return the underlying WhisperModel instance."""
        if self._model is not None:
            return self._model

        # If it's a Path object, check if it exists locally
        if isinstance(self.model_path, Path) and not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Available local models: {list(MODEL_REGISTRY.keys())}"
            )

        try:
            self._model = WhisperModel(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            logger.info("Loaded FasterWhisper model on %s (%s)", self.device, self.compute_type)
        except Exception as exc:
            if self.device == "cuda":
                logger.warning(
                    "Failed to load FasterWhisper on CUDA (%s). Falling back to CPU int8...",
                    exc,
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    str(self.model_path),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                )
            else:
                raise

        return self._model

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe an audio array or file.

        Returns a dictionary containing:
          - text: Combined full transcript text string
          - segments: List of segment dictionaries with start, end, text
          - language: Detected language code
          - language_probability: Probability of detected language
          - duration: Audio duration in seconds
        """
        model = self.load()

        if isinstance(audio, (str, Path)):
            audio_path = Path(audio).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            import soundfile as sf
            audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if sr != 16000:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=int(sr), target_sr=16000).astype(np.float32)
                except ImportError:
                    import scipy.signal
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
            audio_input = audio_data
        else:
            audio_input = audio

        segments_gen, info = model.transcribe(
            audio_input,
            language=language,
            beam_size=beam_size,
            **kwargs,
        )

        segments = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments_gen
        ]
        full_text = " ".join(s["text"].strip() for s in segments)

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

