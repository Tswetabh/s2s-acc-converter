"""Regression coverage for opt-in GPU-only Medium model loading."""

import unittest
from unittest.mock import patch

from ASR.asr_validator import load_asr_model_adaptive


class StrictGpuAsrLoaderTests(unittest.TestCase):
    """Verify strict mode cannot quietly change the Medium verifier to CPU."""

    def test_strict_gpu_load_never_attempts_cpu_fallback(self) -> None:
        """Return a clear GPU-load result after the first CUDA construction fails."""
        with patch(
            "faster_whisper.WhisperModel",
            side_effect=RuntimeError("CUDA out of memory"),
        ) as whisper_model:
            model, device = load_asr_model_adaptive(
                "medium",
                force_device="cuda",
                allow_cpu_fallback=False,
            )

        self.assertIsNone(model)
        self.assertEqual(device, "cuda_load_failed")
        self.assertEqual(whisper_model.call_count, 1)
        self.assertEqual(whisper_model.call_args.kwargs["device"], "cuda")
