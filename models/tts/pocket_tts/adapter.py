"""Pocket TTS Model Integration Adapter.

Provides a clean interface for loading and running the Pocket TTS model
via the core pocket_tts package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "tts" / "pocket_tts"


class PocketTTSAdapter:
    """Adapter for Pocket TTS inference and voice cloning."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self._model = None

    def load(self, variant: Optional[str] = None):
        """Load and return the underlying TTSModel instance."""
        if self._model is not None:
            return self._model

        import sys
        engine_dir = REPO_ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
        if str(engine_dir) not in sys.path:
            sys.path.insert(0, str(engine_dir))

        from pocket_tts.models.tts_model import TTSModel
        kwargs = {"device": self.device}
        if variant:
            kwargs["variant"] = variant

        self._model = TTSModel.load_model(**kwargs)
        return self._model

    def get_voice_state(self, prompt: Union[str, Path]):
        """Compute conditioning state for custom reference voice or preset name."""
        model = self.load()
        return model.get_state_for_audio_prompt(str(prompt))

    def generate(self, voice_state: Any, text: str) -> torch.Tensor:
        """Generate audio tensor for given text."""
        model = self.load()
        return model.generate_audio(voice_state, text_to_generate=text)
