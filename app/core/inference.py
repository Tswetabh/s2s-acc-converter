"""Inference lifecycle manager enforcing single-model residency and memory cleanup."""

import time
import os
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.base_tts import BaseTTSModel
from app.core.gpu_monitor import GPUMonitor
from app.core.audio_utils import get_audio_metadata
from app.exceptions import ModelExecutionError, HardwareIncompatibleError
from app.config import HW_CLOUD_RECOMMENDED

class InferenceEngine:
    """Manages strictly isolated, single-model TTS execution with rigorous resource tracking."""

    _active_model: Optional[BaseTTSModel] = None

    @classmethod
    def unload_active_model(cls) -> None:
        """Safely unload whatever model is currently holding VRAM."""
        if cls._active_model is not None:
            try:
                cls._active_model.unload()
            except Exception:
                pass
            cls._active_model = None
        GPUMonitor.cleanup()

    @classmethod
    def run_generation(
        cls,
        model_adapter: BaseTTSModel,
        text: str,
        output_path: str,
        device: str = "auto",
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        reference_audio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute inference with strict memory lifecycle, timing, and error isolation."""
        
        # 1. Ensure clean slate - unload any other model
        if cls._active_model is not None and cls._active_model is not model_adapter:
            cls.unload_active_model()
        else:
            GPUMonitor.cleanup()

        GPUMonitor.reset_peak_stats()
        mem_before = GPUMonitor.capture_state()
        vram_before = mem_before["vram_used_gb"]
        ram_before = mem_before["ram_used_gb"]

        load_time_sec = 0.0
        # 2. Load model if not loaded
        if not model_adapter.is_loaded:
            t0_load = time.perf_counter()
            model_adapter.load(device=device)
            load_time_sec = round(time.perf_counter() - t0_load, 3)
            cls._active_model = model_adapter

        # 3. Generate Audio under inference context
        t0_gen = time.perf_counter()
        try:
            gen_result = model_adapter.generate(
                text=text,
                output_path=output_path,
                language=language,
                speaker=speaker,
                reference_audio=reference_audio,
                **kwargs
            )
        except Exception as e:
            cls.unload_active_model()
            raise ModelExecutionError(f"Generation failed: {str(e)}") from e

        generation_time_sec = round(time.perf_counter() - t0_gen, 3)

        # 4. Measure Audio metrics
        audio_meta = get_audio_metadata(output_path)
        audio_dur = audio_meta["duration_sec"]
        rtf = round(generation_time_sec / audio_dur, 4) if audio_dur > 0 else 0.0

        # 5. Measure peak memory and memory after
        vram_peak = GPUMonitor.get_peak_vram_gb(vram_before)
        mem_after = GPUMonitor.capture_state()
        vram_after = mem_after["vram_used_gb"]
        ram_peak = max(ram_before, mem_after["ram_used_gb"])

        return {
            "output_path": output_path,
            "load_time_sec": load_time_sec,
            "generation_time_sec": generation_time_sec,
            "audio_duration_sec": audio_dur,
            "rtf": rtf,
            "ttfa_sec": gen_result.get("ttfa_sec", "N/A"),
            "vram_before_gb": vram_before,
            "vram_peak_gb": vram_peak,
            "vram_after_gb": vram_after,
            "ram_before_gb": ram_before,
            "ram_peak_gb": ram_peak,
            "sample_rate": audio_meta["sample_rate"],
            "file_size_mb": audio_meta["file_size_mb"],
            "status": "SUCCESS",
            "error": "",
        }
