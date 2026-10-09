"""Unit and integration test suite for Unified TTS POC platform."""

import unittest
import os
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.registry import get_all_models, get_model, list_available_models, MODELS
from app.system_info import get_system_environment
from app.core.gpu_monitor import GPUMonitor
from app.core.audio_utils import get_audio_metadata
from app.core.benchmark import BenchmarkEngine
from app.core.result_store import ResultStore
from app.core.scoring import ScoringEngine

class TestTTSFramework(unittest.TestCase):
    """Test suite verifying all acceptance criteria and platform components."""

    def test_model_registry_contains_all_14_models(self):
        """Verify all 14 requested model adapters are registered."""
        expected_models = [
            "kokoro", "styletts2", "cosyvoice2", "cosyvoice3", "f5tts",
            "piper", "qwen3tts", "chatterbox", "xtts", "orpheus",
            "megatts3", "dia2", "vibevoice", "fishs2"
        ]
        self.assertEqual(len(MODELS), 14)
        for m_id in expected_models:
            self.assertIn(m_id, MODELS, f"Model {m_id} missing from registry")
            adapter = get_model(m_id)
            self.assertIsNotNone(adapter, f"Failed to instantiate {m_id}")
            meta = adapter.get_metadata()
            self.assertTrue(len(meta["name"]) > 0)
            self.assertIn("hardware_classification", meta)
            self.assertIn("license_code", meta)
            self.assertIn("license_weights", meta)

    def test_health_checks_provide_structured_diagnostics(self):
        """Verify health check never throws and returns correct keys."""
        models = get_all_models()
        self.assertEqual(len(models), 14)
        for m in models:
            health = m.health_check()
            self.assertIn("model", health)
            self.assertIn("installed", health)
            self.assertIn("weights_available", health)
            self.assertIn("cuda_available", health)
            self.assertIn("free_vram_gb", health)
            self.assertIn("status", health)
            self.assertIn("reason", health)

    def test_evaluation_texts_contain_all_required_categories(self):
        """Verify evaluation_texts.json has all 12 categories including long paragraph."""
        tests = BenchmarkEngine.load_evaluation_texts()
        self.assertGreaterEqual(len(tests), 12)
        categories = {t["category"] for t in tests}
        required_cats = {
            "normal", "conversation", "question", "numbers", "dates",
            "technical", "expressive", "punctuation", "proper_nouns",
            "long_form", "hindi", "hinglish"
        }
        for cat in required_cats:
            self.assertIn(cat, categories, f"Category '{cat}' missing from evaluation_texts.json")

    def test_gpu_monitoring_lifecycle(self):
        """Verify memory capture and cleanup methods execute safely."""
        state = GPUMonitor.capture_state()
        self.assertIn("vram_used_gb", state)
        self.assertIn("vram_free_gb", state)
        self.assertIn("ram_used_gb", state)
        # Cleanup should never crash
        GPUMonitor.cleanup()

    def test_system_info_detection(self):
        """Verify hardware detection runs without errors."""
        env = get_system_environment()
        self.assertIn("cpu", env)
        self.assertIn("vram_total_gb", env)
        self.assertIn("ram_total_gb", env)
        self.assertIn("os", env)

    def test_scoring_engine_and_rankings(self):
        """Verify composite scoring and ranking derivation."""
        scorer = ScoringEngine()
        bench_df = ResultStore.load_benchmark_results()
        human_df = ResultStore.load_human_scores()
        meta_map = {m.get_metadata()["model_id"]: m.get_metadata() for m in get_all_models()}
        
        comp_df = scorer.compute_composite_scores(bench_df, human_df, meta_map)
        self.assertEqual(len(comp_df), 14)
        rankings = scorer.derive_rankings(comp_df)
        self.assertIn("best_naturalness", rankings)
        self.assertIn("best_local_3050", rankings)
        self.assertIn("best_low_latency", rankings)

if __name__ == "__main__":
    unittest.main()
