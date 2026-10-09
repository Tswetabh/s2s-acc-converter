# Configuration system for audiobook generation

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
try:
    from ..preprocessing.schema import Config
except ImportError:
    # Fallback for testing
    from preprocessing.schema import Config

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

class ConfigManager:
    """Manages configuration loading and validation."""

    DEFAULT_CONFIG = {
        # AUDIOBOOK GENERATION CONFIGURATION

        # CHUNKING SETTINGS - Used in: SmartChunker class
        'chunking': {
            'mode': 'paragraph',  # "sentence" or "paragraph"
            'min_words': 35,      # Minimum words per chunk (GPU-optimized)
            'target_words': 50,    # Target for sentence-bounded word_target mode
            'respect_boundaries': True
        },

        # DEVICE SETTINGS - Used in: TTSModel, AudiobookGenerator
        'device': {
            'auto': True,             # Auto-detect CUDA if available
            'preferred': 'cuda',      # "cpu", "cuda", or "auto"
            'max_nb_tokens_in_a_chunk': 50  # Internal token limit per chunk
        },

        # TTS CORE PARAMETERS - Used in: TTSModel class
        'tts_core': {
            'temperature': 0.85,           # Sampling temperature (0.0-1.5)
            'eos_threshold': -3.0,        # End-of-sequence threshold
            'frames_after_eos': 2         # Frames to generate after EOS detection
        },

        # AUDIO ENDPOINT CLEANUP - Used in: audiobook post-processing
        'audio_cleanup': {
            'enabled': True,
            'use_silero_vad': True,
            'speech_endpoint_threshold': 0.004,
            'trimming_buffer_ms': 100,
            # Overlap Silero/RMS + WAV write with next GPU batch (per-worker thread).
            'async': True,
            'max_pending': 12,
        },

        # EMOTION ANALYSIS - Used in: EmotionAnalyzer class
        'emotion': {
            'enabled': True,
            'model': 'j-hartmann/emotion-english-distilroberta-base',
            'mappings': {
                'joy': {'temperature': 0.90, 'speed_factor': 1.00},
                'surprise': {'temperature': 0.85, 'speed_factor': 1.10},
                'anger': {'temperature': 0.80, 'speed_factor': 1.10},
                'neutral': {'temperature': 0.70, 'speed_factor': 1.0},
                'sadness': {'temperature': 0.65, 'speed_factor': 0.85},
                'fear': {'temperature': 0.60, 'speed_factor': 0.90},
                'disgust': {'temperature': 0.65, 'speed_factor': 0.95}
            },
            'keyword_boosts': {
                'joy': {
                    'keywords': ['ecstatic', 'thrilled', 'delighted'],
                    'temperature_boost': 0.03,
                    'speed_boost': 0.02
                },
                'anger': {
                    'keywords': ['enraged', 'infuriated', 'outraged'],
                    'temperature_boost': 0.05,
                    'speed_boost': 0.03
                }
            }
        },

        # PAUSE DURATIONS - Used in: ParameterMapper.calculate_pause()
        'pauses': {
            'base_durations': {
                'sentence_end': 160,      # ms
                'paragraph_break': 320,   # ms
                'chapter_start': 800,     # ms
                'question_mark': 240,     # ms
                'exclamation': 80,        # ms
                'ellipsis': 320           # ms
            },
            'emotion_multipliers': {
                'anger': 1.3,     # Angry pauses are 30% longer
                'fear': 1.2,      # Fearful pauses 20% longer
                'sadness': 1.4,   # Sad pauses 40% longer
                'joy': 0.9,       # Joyful pauses 10% shorter
                'neutral': 1.0    # Baseline
            }
        },

        # PAUSE INJECTION - Used in: pause_injector.inject_pauses_for_punctuation()
        'pause_injection': {
            'enabled': True,
            'punctuation_durations': {
                '.': 0.50,
                '!': 0.25,
                '?': 0.35,
                ',': 0.0,
                '...': 0.40,
                '--': 0.25,
                ';': 0.20,
                ':': 0.20,
            }
        },

        # SPEED VARIATION - Used in: ParameterMapper
        'speed_variation': {
            'enabled': True,                    # Toggle on/off
            'method': 'librosa',               # "librosa" or "scipy"
            'range': 0.10,                     # ±10% variance
            'emotion_speed_modifiers': {
                'joy': 0.15,
                'surprise': 0.10,
                'anger': 0.10,
                'sadness': -0.15,
                'fear': -0.10,
                'disgust': -0.05,
                'neutral': 0.0
            }
        },

        # M4B / MP3 / WAV export — only requested formats are built
        'm4b': {
            'write_m4b': False,
            'write_mp3': False,
            'write_wav': True,
            'enabled': False,                  # legacy: any of write_* true
            'speed': 1.0,
            'sample_rate': 24000,
            'normalization_type': 'peak',
            'target_db': -1.5,
            'chapterize': False,
            'max_chapter_minutes': 0,
        },

        # SHORT SENTENCE MITIGATION - Used in: ParameterMapper class
        'mitigation': {
            'short_sentence_threshold': 5,        # Max words to trigger mitigation
            'temperature_short_sentence': 0.5     # Override temperature for short sentences
        },

        # PARALLEL PROCESSING - Used in: AudiobookGenerator class
        'parallel': {
            'enabled': True,                      # Enable parallel processing
            'max_workers': 6,                     # Sets default in GUI (GPU: each worker loads own model)
            'min_workers': 1,                     # Minimum workers (fallback to sequential)
            'ram_limit_percent': 85,              # RAM usage threshold (%) (GPU shift: less CPU RAM needed)
            'load_threshold': 8.0,                # System load threshold (normalized)
            'worker_vram_gb': 1.0,                 # Estimated VRAM per worker model
            'vram_reserve_percent': 10.0,          # VRAM kept free during worker startup
            'vram_startup_reserve_percent': 5.0,   # Reserve for transient worker startup peaks
            'adaptive_workers': False             # Adaptive worker count (disabled for testing)
        },

        # BATCHED GENERATION - Opt-in until quality and throughput are validated
        'batch_generation': {
            'enabled': False,
            'batch_size': 2,
            'max_padding_ratio': 0.0,
            'batch_regeneration': False,
        },

        # CUDA graphs for FlowLM AR (opt-in; CUDA only)
        'cuda_graphs': {
            'enabled': False,
            'fallback_eager': True,
        },

        # ASR QUALITY CONTROL - Used in: ASR integration
        'asr_quality_control': {
            'enabled': False,                     # Enable ASR quality control
            'pipeline': 'legacy',                  # Single two-stage flow; kept for old saved configs
            'second_stage_model': 'medium',         # Stage 2 verifier; 'disabled' skips verification
            'alignment_diagnostic_enabled': False, # Unused in the single-flow GUI
            'engine': 'faster_whisper',            # legacy default backend
            'model': 'base',
            'language': 'en',
            'asr_threshold': 0.75,
            'max_retries': 3,
            'temp_decrement': 0.1,
            'regeneration_strategy': 'staged_three_candidates',
            'regen_tts_workers': 2,
            'regen_tts_batch_size': 2,
            'regen_worker_start_gap_s': 2.0,
            'device_policy': 'gpu_after_tts',
            'parallel': {
                'cpu_workers': 0,
                'gpu_workers': 0,
                # CUDA workers after TTS gen only (1–8). 0 = skip post-gen ASR.
                'gpu_workers_after_tts': 4,
                'n_threads': 0,
            },
            # done=false until first auto-cal; then true (never auto-run again)
            'worker_calibration': {
                'done': False,
                'recommended': None,
                'last_run': None,
            },
        },

        # GUI SETTINGS - Used in: MainWindow class
        'gui': {
            'theme': 'system',          # "light", "dark", or "system"
            'auto_save_interval': 30,   # Seconds between progress saves
            'max_preview_chunks': 50,   # How many chunks to show in preview
            'default_output_format': 'wav'  # "wav" or "mp3"
        },

        # DEBUG & DEVELOPMENT SETTINGS
        'debug': {
            'log_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR
            'save_intermediate': False, # Save temp files for debugging
            'profile_performance': False # Enable performance profiling
        },

        # QUALITY SETTINGS
        'quality': {
            'lsd_steps': 2  # Lagrangian Self Distillation steps (1-30; 5-10 optimal)
        }
    }

    @classmethod
    def load_config(cls, config_path: Optional[str] = "pocket_tts/config/default_config.yaml") -> Config:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to config file, or None for defaults

        Returns:
            Config: Validated configuration object
        """
        config_data = cls.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            try:
                if HAS_YAML and config_path.endswith(('.yaml', '.yml')):
                    with open(config_path, 'r') as f:
                        user_config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    with open(config_path, 'r') as f:
                        user_config = json.load(f)
                else:
                    # Try JSON first, then YAML if available
                    try:
                        with open(config_path, 'r') as f:
                            user_config = json.load(f)
                    except:
                        if HAS_YAML:
                            with open(config_path, 'r') as f:
                                user_config = yaml.safe_load(f)
                        else:
                            raise ValueError("Could not parse config file")

                # Merge with defaults
                config_data = cls._merge_configs(config_data, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
                print("Using default configuration.")

        # Validate and create Config object
        try:
            config = Config(**config_data)

            # --- START Validation for max_workers ---
            import psutil
            physical_cores = psutil.cpu_count(logical=False) if psutil else None

            if physical_cores:
                max_workers_set = config.parallel.get('max_workers', 0)
                if max_workers_set > physical_cores:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Configured max_workers ({max_workers_set}) exceeds system physical cores ({physical_cores}). "
                        "The system may become unresponsive. It is recommended to leave at least one core free."
                    )
            # --- END Validation for max_workers ---

            # Store the config path for saving
            config._config_path = config_path
            return config
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}")

    @classmethod
    def save_default_config(cls, output_path: str) -> None:
        """
        Save the default configuration to a file.

        Args:
            output_path: Where to save the config file
        """
        if HAS_YAML and output_path.endswith(('.yaml', '.yml')):
            with open(output_path, 'w') as f:
                yaml.dump(cls.DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        elif output_path.endswith('.json'):
            # JSON format
            with open(output_path, 'w') as f:
                json.dump(cls.DEFAULT_CONFIG, f, indent=2)
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(cls.DEFAULT_CONFIG, f, indent=2)

        # Add header comment if it's a file that supports comments
        if output_path.endswith(('.yaml', '.yml')) or (not HAS_YAML and output_path.endswith('.json')):
            with open(output_path, 'r') as f:
                content = f.read()

            header = """
# =====================================================================
# PocketTTS Configuration File
# =====================================================================
#
# This file contains all configuration settings for PocketTTS.
# Modify values here to customize behavior.
#
# Units:
# - Durations: milliseconds (ms)
# - Sample rates: Hz
# - Quality levels: 1-4 (higher = better quality, slower)
#
# =====================================================================

"""
            with open(output_path, 'w') as f:
                f.write(header + content)

    @classmethod
    def _merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge override config into base config.

        Args:
            base: Base configuration
            override: User overrides

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge dictionaries
                result[key] = cls._merge_configs(result[key], value)
            else:
                # Override or add new value
                result[key] = value

        return result

    @classmethod
    def get_config_paths(cls) -> List[str]:
        """
        Get possible config file locations in order of priority.

        Returns:
            List of possible config file paths
        """
        paths = []

        # Current directory
        paths.append('./audiobook_config.yaml')
        paths.append('./config.yaml')

        # User's home directory
        home = Path.home()
        paths.append(str(home / '.audiobook_config.yaml'))
        paths.append(str(home / '.config' / 'pocket_tts' / 'config.yaml'))

        return paths

    @classmethod
    def find_config(cls) -> Optional[str]:
        """
        Find the first available config file.

        Returns:
            Path to config file, or None if none found
        """
        for path in cls.get_config_paths():
            if os.path.exists(path):
                return path
        return None
