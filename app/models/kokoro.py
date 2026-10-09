"""Kokoro v1.0 (82M) Model Adapter."""

import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.base_tts import BaseTTSModel
from app.core.audio_utils import save_numpy_to_wav
from app.system_info import get_gpu_info
from app.config import (
    HW_LOCAL_COMFORTABLE,
    STATUS_READY,
    STATUS_DEPENDENCY_MISSING,
    STATUS_WEIGHTS_MISSING,
    STATUS_CUDA_UNAVAILABLE,
)
from app.exceptions import DependencyMissingError, ModelExecutionError

class KokoroModel(BaseTTSModel):
    """Adapter for hexgrad's Kokoro-82M model."""

    name: str = "Kokoro v1.0 — 82M"
    model_id: str = "kokoro"
    version: str = "1.0"
    parameter_count: str = "82M"
    languages: List[str] = ["en", "hi", "fr", "ja", "it", "es", "pt", "zh", "hi-en"]
    supports_voice_cloning: bool = False
    supports_multispeaker: bool = True
    supports_streaming: bool = True
    streaming_type: str = "chunked"
    cloning_type: str = "predefined"
    sample_rate: int = 24000
    license_code: str = "Apache 2.0"
    license_weights: str = "Apache 2.0"
    repo_url: str = "https://github.com/hexgrad/kokoro"
    paper_url: str = "https://huggingface.co/hexgrad/Kokoro-82M"
    expected_vram_gb: float = 1.2
    hardware_classification: str = HW_LOCAL_COMFORTABLE
    official_reported_rtf: Optional[float] = 0.20
    official_reported_ttfa_ms: Optional[float] = 180.0

    # Default speaker choices
    DEFAULT_VOICE = "af_bella"

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def health_check(self) -> Dict[str, Any]:
        """Diagnose dependencies, weights, and CUDA availability for Kokoro."""
        gpu = get_gpu_info()
        installed = False
        reason = ""

        try:
            import kokoro
            installed = True
        except ImportError as e:
            reason = f"Python package 'kokoro' not installed: {str(e)}"

        status = STATUS_READY
        if not installed:
            status = STATUS_DEPENDENCY_MISSING
        elif not gpu["available"]:
            status = STATUS_CUDA_UNAVAILABLE
            reason = "CUDA GPU not available; will execute on CPU."

        return {
            "model": self.name,
            "installed": installed,
            "weights_available": True,  # Downloaded on first load automatically by pipeline
            "cuda_available": gpu["available"],
            "free_vram_gb": gpu["free_vram_gb"],
            "status": status,
            "reason": reason or "Kokoro v1.0 is ready for inference.",
        }

    def load(self, device: str = "auto") -> None:
        """Load Kokoro KPipeline onto resolved device."""
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise DependencyMissingError("Package 'kokoro' is missing. Run: pip install kokoro soundfile misaki") from e

        resolved_device = self.determine_device(device)
        self.device = resolved_device

        try:
            # English pipeline by default, can switch lang_code per call
            self.pipeline = KPipeline(lang_code="a", device=resolved_device)
            self.is_loaded = True
        except Exception as e:
            # Fallback to cpu if cuda device fails
            if resolved_device == "cuda":
                self.device = "cpu"
                self.pipeline = KPipeline(lang_code="a", device="cpu")
                self.is_loaded = True
            else:
                raise ModelExecutionError(f"Failed to load Kokoro pipeline: {str(e)}") from e

    def generate(
        self,
        text: str,
        output_path: str,
        language: Optional[str] = "en",
        speaker: Optional[str] = None,
        reference_audio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate audio using Kokoro pipeline."""
        if not self.is_loaded or self.pipeline is None:
            self.load()

        voice = speaker or self.DEFAULT_VOICE
        # Infer lang_code directly from voice prefix if it matches Kokoro convention (e.g., af_..., bm_..., hf_...)
        if voice and len(voice) > 2 and voice[0] in ["a", "b", "h", "e", "f", "j", "i", "z"] and voice[1] in ["f", "m"]:
            lang_code = voice[0]
        elif language in ["en", "en-us"]:
            lang_code = "a"
        elif language in ["en-gb", "gb"]:
            lang_code = "b"
        elif language in ["hi", "hi-en"]:
            lang_code = "h"
        elif language in ["es"]:
            lang_code = "e"
        elif language in ["fr"]:
            lang_code = "f"
        elif language in ["ja"]:
            lang_code = "j"
        elif language in ["zh"]:
            lang_code = "z"
        elif language in ["it"]:
            lang_code = "i"
        else:
            lang_code = "a"

        # Update pipeline language if different
        if hasattr(self.pipeline, "lang_code") and self.pipeline.lang_code != lang_code:
            try:
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code=lang_code, device=self.device)
            except Exception:
                pass

        t0 = time.perf_counter()
        ttfa_sec = None
        all_audio_chunks = []

        try:
            generator = self.pipeline(text, voice=voice, speed=kwargs.get("speed", 1.0), split_pattern=r"\n+")
            for i, (gs, ps, audio) in enumerate(generator):
                if i == 0:
                    ttfa_sec = round(time.perf_counter() - t0, 3)
                if audio is not None:
                    all_audio_chunks.append(audio)
        except Exception as e:
            # If specified voice doesn't match lang_code, fallback to default voice
            try:
                generator = self.pipeline(text, voice=self.DEFAULT_VOICE, speed=1.0)
                for i, (gs, ps, audio) in enumerate(generator):
                    if i == 0:
                        ttfa_sec = round(time.perf_counter() - t0, 3)
                    if audio is not None:
                        all_audio_chunks.append(audio)
            except Exception as e2:
                raise ModelExecutionError(f"Kokoro generation error: {str(e2)}") from e2

        if not all_audio_chunks:
            raise ModelExecutionError("Kokoro produced empty audio stream.")

        import numpy as np
        full_audio = np.concatenate(all_audio_chunks, axis=0)
        saved_path = save_numpy_to_wav(full_audio, output_path, sample_rate=self.sample_rate)

        gen_time = round(time.perf_counter() - t0, 3)
        audio_dur = round(len(full_audio) / float(self.sample_rate), 3)

        return {
            "output_path": saved_path,
            "audio_duration_sec": audio_dur,
            "generation_time_sec": gen_time,
            "sample_rate": self.sample_rate,
            "ttfa_sec": ttfa_sec or gen_time,
            "streaming_chunks": len(all_audio_chunks),
        }

    def unload(self) -> None:
        """Unload Kokoro pipeline and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        super().unload()
