"""Common abstract base class for all Text-to-Speech model adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import gc
from pathlib import Path

from app.config import (
    HW_LOCAL_COMFORTABLE,
    HW_LOCAL_MEMORY_LIMITED,
    HW_CLOUD_RECOMMENDED,
    HW_NOT_RUNNABLE,
    STATUS_READY,
    STATUS_MEMORY_LIMITED,
    STATUS_DEPENDENCY_MISSING,
    STATUS_WEIGHTS_MISSING,
    STATUS_CUDA_UNAVAILABLE,
    STATUS_CLOUD_RECOMMENDED,
    STATUS_INCOMPATIBLE,
)
from app.system_info import get_gpu_info

class BaseTTSModel(ABC):
    """Abstract Base Class conforming to unified TTS POC requirements."""

    name: str = ""
    model_id: str = ""
    version: str = ""
    parameter_count: str = ""
    languages: List[str] = ["en"]
    supports_voice_cloning: bool = False
    supports_multispeaker: bool = False
    supports_streaming: bool = False
    streaming_type: str = "not_supported"  # 'native', 'experimental', 'chunked', 'not_supported'
    cloning_type: str = "none"  # 'zero_shot', 'predefined', 'adaptation', 'none'
    sample_rate: int = 24000
    license_code: str = "Unknown"
    license_weights: str = "Unknown"
    repo_url: str = ""
    paper_url: str = ""
    expected_vram_gb: float = 2.0
    hardware_classification: str = HW_LOCAL_COMFORTABLE
    official_reported_rtf: Optional[float] = None
    official_reported_ttfa_ms: Optional[float] = None

    def __init__(self):
        self.is_loaded: bool = False
        self.device: str = "cpu"
        self.model: Any = None

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform diagnostics on dependency, weights, and hardware compatibility.

        Returns:
            Dict containing:
                model (str), installed (bool), weights_available (bool),
                cuda_available (bool), free_vram_gb (float), status (str),
                reason (str)
        """
        pass

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load the model weights into memory/VRAM.
        
        Args:
            device: 'auto', 'cuda', 'cpu', or device ID
        """
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        reference_audio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Synthesize audio for the given text.

        Returns:
            Dict containing:
                output_path (str), audio_duration_sec (float),
                generation_time_sec (float), sample_rate (int),
                ttfa_sec (Optional[float]), streaming_chunks (int)
        """
        pass

    def unload(self) -> None:
        """Unload model from memory and free GPU VRAM completely."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return standardized metadata card for this model."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "version": self.version,
            "parameter_count": self.parameter_count,
            "languages": self.languages,
            "supports_voice_cloning": self.supports_voice_cloning,
            "supports_multispeaker": self.supports_multispeaker,
            "supports_streaming": self.supports_streaming,
            "streaming_type": self.streaming_type,
            "cloning_type": self.cloning_type,
            "sample_rate": self.sample_rate,
            "license_code": self.license_code,
            "license_weights": self.license_weights,
            "repo_url": self.repo_url,
            "paper_url": self.paper_url,
            "expected_vram_gb": self.expected_vram_gb,
            "hardware_classification": self.hardware_classification,
            "official_reported_rtf": self.official_reported_rtf,
            "official_reported_ttfa_ms": self.official_reported_ttfa_ms,
        }

    def determine_device(self, requested: str = "auto") -> str:
        """Resolve requested device against system capability and hardware classification."""
        if requested != "auto":
            return requested
        gpu = get_gpu_info()
        if gpu["available"] and self.hardware_classification != HW_CLOUD_RECOMMENDED:
            return "cuda"
        return "cpu"
