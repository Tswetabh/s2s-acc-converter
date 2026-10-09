"""Result persistence and retrieval for benchmarks and human evaluations."""

import os
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from app.config import RESULTS_CSV_PATH, HUMAN_SCORES_CSV_PATH

BENCHMARK_COLUMNS = [
    "timestamp",
    "model",
    "model_version",
    "test_case",
    "device",
    "dtype",
    "load_time_sec",
    "generation_time_sec",
    "audio_duration_sec",
    "rtf",
    "ttfa_sec",
    "vram_before_gb",
    "vram_peak_gb",
    "vram_after_gb",
    "ram_before_gb",
    "ram_peak_gb",
    "sample_rate",
    "file_size_mb",
    "status",
    "error",
]

HUMAN_EVAL_COLUMNS = [
    "timestamp",
    "evaluator",
    "sample_id",
    "model",
    "test_case",
    "naturalness",
    "pronunciation",
    "prosody",
    "expressiveness",
    "speaker_similarity",
    "overall_quality",
    "notes",
]

class ResultStore:
    """Manages appending and reading benchmark records and human evaluations."""

    @staticmethod
    def init_storage():
        """Ensure CSV files exist with appropriate headers."""
        RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not RESULTS_CSV_PATH.exists() or os.path.getsize(RESULTS_CSV_PATH) == 0:
            with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(BENCHMARK_COLUMNS)

        HUMAN_SCORES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not HUMAN_SCORES_CSV_PATH.exists() or os.path.getsize(HUMAN_SCORES_CSV_PATH) == 0:
            with open(HUMAN_SCORES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(HUMAN_EVAL_COLUMNS)

    @classmethod
    def append_benchmark_result(cls, record: Dict[str, Any]) -> None:
        """Append a completed benchmark measurement to results.csv."""
        cls.init_storage()
        row = [
            record.get("timestamp", datetime.now().isoformat()),
            record.get("model", ""),
            record.get("model_version", ""),
            record.get("test_case", ""),
            record.get("device", ""),
            record.get("dtype", ""),
            record.get("load_time_sec", 0.0),
            record.get("generation_time_sec", 0.0),
            record.get("audio_duration_sec", 0.0),
            record.get("rtf", 0.0),
            record.get("ttfa_sec", "N/A"),
            record.get("vram_before_gb", 0.0),
            record.get("vram_peak_gb", 0.0),
            record.get("vram_after_gb", 0.0),
            record.get("ram_before_gb", 0.0),
            record.get("ram_peak_gb", 0.0),
            record.get("sample_rate", 0),
            record.get("file_size_mb", 0.0),
            record.get("status", "SUCCESS"),
            record.get("error", ""),
        ]
        with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @classmethod
    def append_human_eval(cls, record: Dict[str, Any]) -> None:
        """Append a human evaluation score to human_scores.csv."""
        cls.init_storage()
        row = [
            record.get("timestamp", datetime.now().isoformat()),
            record.get("evaluator", "anonymous"),
            record.get("sample_id", ""),
            record.get("model", ""),
            record.get("test_case", ""),
            record.get("naturalness", 3),
            record.get("pronunciation", 3),
            record.get("prosody", 3),
            record.get("expressiveness", 3),
            record.get("speaker_similarity", 3),
            record.get("overall_quality", 3),
            record.get("notes", ""),
        ]
        with open(HUMAN_SCORES_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @classmethod
    def load_benchmark_results(cls) -> pd.DataFrame:
        """Load benchmark results as pandas DataFrame."""
        cls.init_storage()
        try:
            return pd.read_csv(RESULTS_CSV_PATH)
        except Exception:
            return pd.DataFrame(columns=BENCHMARK_COLUMNS)

    @classmethod
    def load_human_scores(cls) -> pd.DataFrame:
        """Load human evaluation scores as pandas DataFrame."""
        cls.init_storage()
        try:
            return pd.read_csv(HUMAN_SCORES_CSV_PATH)
        except Exception:
            return pd.DataFrame(columns=HUMAN_EVAL_COLUMNS)
