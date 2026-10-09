"""Unified Model Registry for all 14 Text-to-Speech Model Adapters."""

from typing import Dict, Type, List, Optional
from app.core.base_tts import BaseTTSModel

# Import all 14 Model Adapters
from app.models.kokoro import KokoroModel
from app.models.styletts2 import StyleTTS2Model
from app.models.cosyvoice2 import CosyVoice2Model
from app.models.cosyvoice3 import CosyVoice3Model
from app.models.f5tts import F5TTSModel
from app.models.piper import PiperModel
from app.models.qwen3tts import Qwen3TTSModel
from app.models.chatterbox import ChatterboxModel
from app.models.xtts import XTTSModel
from app.models.orpheus import OrpheusModel
from app.models.megatts3 import MegaTTS3Model
from app.models.dia2 import Dia2Model
from app.models.vibevoice import VibeVoiceModel
from app.models.fishs2 import FishS2Model

# Section 11: Model Registry Mapping
MODELS: Dict[str, Type[BaseTTSModel]] = {
    "kokoro": KokoroModel,
    "styletts2": StyleTTS2Model,
    "cosyvoice2": CosyVoice2Model,
    "cosyvoice3": CosyVoice3Model,
    "f5tts": F5TTSModel,
    "piper": PiperModel,
    "qwen3tts": Qwen3TTSModel,
    "chatterbox": ChatterboxModel,
    "xtts": XTTSModel,
    "orpheus": OrpheusModel,
    "megatts3": MegaTTS3Model,
    "dia2": Dia2Model,
    "vibevoice": VibeVoiceModel,
    "fishs2": FishS2Model,
}

_INSTANCES: Dict[str, BaseTTSModel] = {}

def get_model(model_id: str) -> Optional[BaseTTSModel]:
    """Retrieve or instantiate a singleton adapter for the given model_id."""
    clean_id = model_id.lower().replace("-", "").replace("_", "")
    for k, cls in MODELS.items():
        if k == clean_id or k == model_id.lower():
            if k not in _INSTANCES:
                _INSTANCES[k] = cls()
            return _INSTANCES[k]
    return None

def get_all_models() -> List[BaseTTSModel]:
    """Return initialized instances of all 14 registered models."""
    res = []
    for k, cls in MODELS.items():
        if k not in _INSTANCES:
            _INSTANCES[k] = cls()
        res.append(_INSTANCES[k])
    return res

def list_available_models() -> List[str]:
    """Return keys for all registered model identifiers."""
    return list(MODELS.keys())
