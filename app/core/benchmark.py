"""Standardized benchmarking engine for TTS models."""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.core.base_tts import BaseTTSModel
from app.core.inference import InferenceEngine
from app.core.result_store import ResultStore
from app.core.audio_utils import get_audio_metadata
from app.config import (
    EVALUATION_TEXTS_PATH,
    AUDIO_OUTPUT_DIR,
    DEFAULT_REFERENCE_AUDIO,
    HW_CLOUD_RECOMMENDED,
)

class BenchmarkEngine:
    """Executes standardized test suites against TTS model adapters."""

    @staticmethod
    def load_evaluation_texts() -> List[Dict[str, Any]]:
        """Load test cases from data/evaluation_texts.json."""
        if not EVALUATION_TEXTS_PATH.exists():
            return []
        with open(EVALUATION_TEXTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("test_cases", [])

    @classmethod
    def benchmark_model(
        cls,
        model_adapter: BaseTTSModel,
        test_case_ids: Optional[List[str]] = None,
        device: str = "auto",
        skip_cloud_recommended: bool = False,
    ) -> List[Dict[str, Any]]:
        """Benchmark a single model against evaluation test cases."""
        meta = model_adapter.get_metadata()
        m_id = meta["model_id"]
        
        # Check cloud recommendation flag
        if skip_cloud_recommended and meta["hardware_classification"] == HW_CLOUD_RECOMMENDED:
            return [{
                "timestamp": datetime.now().isoformat(),
                "model": m_id,
                "model_version": meta["version"],
                "test_case": "ALL",
                "device": "skipped",
                "dtype": "fp16",
                "load_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
                "ttfa_sec": "N/A",
                "vram_before_gb": 0.0,
                "vram_peak_gb": 0.0,
                "vram_after_gb": 0.0,
                "ram_before_gb": 0.0,
                "ram_peak_gb": 0.0,
                "sample_rate": meta["sample_rate"],
                "file_size_mb": 0.0,
                "status": "SKIPPED_CLOUD_RECOMMENDED",
                "error": "Model classified as CLOUD_RECOMMENDED; skipped in local-only run."
            }]

        all_tests = cls.load_evaluation_texts()
        if test_case_ids:
            all_tests = [t for t in all_tests if t["id"] in test_case_ids]

        results = []
        out_dir = AUDIO_OUTPUT_DIR / m_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for test in all_tests:
            t_id = test["id"]
            t_text = test["text"]
            t_lang = test.get("language", "en")

            # Check language compatibility
            if t_lang not in meta["languages"] and "all" not in meta["languages"] and not (t_lang == "hi-en" and "hi" in meta["languages"]):
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "model": m_id,
                    "model_version": meta["version"],
                    "test_case": t_id,
                    "device": device,
                    "dtype": "fp16",
                    "load_time_sec": 0.0,
                    "generation_time_sec": 0.0,
                    "audio_duration_sec": 0.0,
                    "rtf": 0.0,
                    "ttfa_sec": "N/A",
                    "vram_before_gb": 0.0,
                    "vram_peak_gb": 0.0,
                    "vram_after_gb": 0.0,
                    "ram_before_gb": 0.0,
                    "ram_peak_gb": 0.0,
                    "sample_rate": meta["sample_rate"],
                    "file_size_mb": 0.0,
                    "status": "NOT_SUPPORTED",
                    "error": f"Language '{t_lang}' not supported by {m_id}.",
                }
                ResultStore.append_benchmark_result(rec)
                results.append(rec)
                continue

            out_audio = str(out_dir / f"{m_id}_{t_id}.wav")
            ref_audio = str(DEFAULT_REFERENCE_AUDIO) if meta["supports_voice_cloning"] else None

            try:
                metrics = InferenceEngine.run_generation(
                    model_adapter=model_adapter,
                    text=t_text,
                    output_path=out_audio,
                    device=device,
                    language=t_lang,
                    reference_audio=ref_audio,
                )
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "model": m_id,
                    "model_version": meta["version"],
                    "test_case": t_id,
                    "device": model_adapter.device,
                    "dtype": "fp16" if model_adapter.device == "cuda" else "fp32",
                    "load_time_sec": metrics["load_time_sec"],
                    "generation_time_sec": metrics["generation_time_sec"],
                    "audio_duration_sec": metrics["audio_duration_sec"],
                    "rtf": metrics["rtf"],
                    "ttfa_sec": metrics["ttfa_sec"],
                    "vram_before_gb": metrics["vram_before_gb"],
                    "vram_peak_gb": metrics["vram_peak_gb"],
                    "vram_after_gb": metrics["vram_after_gb"],
                    "ram_before_gb": metrics["ram_before_gb"],
                    "ram_peak_gb": metrics["ram_peak_gb"],
                    "sample_rate": metrics["sample_rate"],
                    "file_size_mb": metrics["file_size_mb"],
                    "status": "SUCCESS",
                    "error": "",
                }
            except Exception as e:
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "model": m_id,
                    "model_version": meta["version"],
                    "test_case": t_id,
                    "device": device,
                    "dtype": "fp16",
                    "load_time_sec": 0.0,
                    "generation_time_sec": 0.0,
                    "audio_duration_sec": 0.0,
                    "rtf": 0.0,
                    "ttfa_sec": "N/A",
                    "vram_before_gb": 0.0,
                    "vram_peak_gb": 0.0,
                    "vram_after_gb": 0.0,
                    "ram_before_gb": 0.0,
                    "ram_peak_gb": 0.0,
                    "sample_rate": meta["sample_rate"],
                    "file_size_mb": 0.0,
                    "status": "FAILED",
                    "error": str(e),
                }

            ResultStore.append_benchmark_result(rec)
            results.append(rec)

        # Unload model after completing benchmark suite to guarantee clean memory
        InferenceEngine.unload_active_model()
        return results
