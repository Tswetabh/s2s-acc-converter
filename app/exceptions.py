"""Custom exceptions for the Unified TTS POC Evaluation Platform."""

class TTSEvaluationError(Exception):
    """Base exception for all TTS evaluation errors."""
    pass

class DependencyMissingError(TTSEvaluationError):
    """Raised when required python packages or native binaries are missing."""
    pass

class WeightsMissingError(TTSEvaluationError):
    """Raised when model weights or checkpoints cannot be found or downloaded."""
    pass

class ModelNotLoadedError(TTSEvaluationError):
    """Raised when an inference operation is requested before loading the model."""
    pass

class HardwareIncompatibleError(TTSEvaluationError):
    """Raised when the target model cannot execute on current hardware."""
    pass

class UnsupportedFeatureError(TTSEvaluationError):
    """Raised when a feature (e.g. voice cloning, streaming) is requested but unsupported."""
    pass

class ModelExecutionError(TTSEvaluationError):
    """Raised when model forward pass or synthesis fails."""
    pass
