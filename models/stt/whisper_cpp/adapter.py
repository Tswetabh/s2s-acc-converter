"""Whisper.cpp Model Integration Adapter.

Provides a clean interface for whisper_cpp integration used by the ASR subsystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "stt" / "whisper_cpp"


class WhisperCppAdapter:
    """Adapter for whisper.cpp / pywhispercpp backend."""

    def __init__(
        self,
        model_name: str = "base.en",
        models_dir: Optional[Union[str, Path]] = None,
        n_threads: int = 4,
    ):
        self.model_name = model_name
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_WEIGHTS_DIR
        self.n_threads = n_threads
        self._backend = None

    def load(self):
        """Initialize whisper.cpp backend using ASR subsystem if available."""
        if self._backend is not None:
            return self._backend

        try:
            import sys
            engine_dir = REPO_ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
            if str(engine_dir) not in sys.path:
                sys.path.insert(0, str(engine_dir))

            from ASR.whisper_cpp_backend import WhisperCppBackend
            self._backend = WhisperCppBackend(
                model_name=self.model_name,
                models_dir=str(self.models_dir),
                n_threads=self.n_threads,
            )
            return self._backend
        except ImportError as exc:
            logger.warning("ASR whisper_cpp_backend could not be imported: %s", exc)
            raise
