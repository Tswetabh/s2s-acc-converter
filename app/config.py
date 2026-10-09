"""Global configuration and paths for Unified TTS POC Evaluation Platform."""

import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_TEXTS_PATH = DATA_DIR / "evaluation_texts.json"
REFERENCE_VOICES_DIR = DATA_DIR / "reference_voices"
DEFAULT_REFERENCE_AUDIO = REFERENCE_VOICES_DIR / "reference_01.wav"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
AUDIO_OUTPUT_DIR = OUTPUTS_DIR / "audio"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
HUMAN_EVAL_DIR = OUTPUTS_DIR / "human_eval"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RESULTS_CSV_PATH = BENCHMARKS_DIR / "results.csv"
HUMAN_SCORES_CSV_PATH = HUMAN_EVAL_DIR / "human_scores.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
DOCS_DIR = PROJECT_ROOT / "docs"

# Force Model & Pip Caches to D: drive
CACHE_DIR = Path("D:/tts-poc-cache")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ensure runtime directories exist
for path in [
    DATA_DIR, REFERENCE_VOICES_DIR, OUTPUTS_DIR, AUDIO_OUTPUT_DIR,
    BENCHMARKS_DIR, HUMAN_EVAL_DIR, REPORTS_DIR, DOCS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)

# Hardware Classifications
HW_LOCAL_COMFORTABLE = "LOCAL — COMFORTABLE"
HW_LOCAL_MEMORY_LIMITED = "LOCAL — MEMORY LIMITED"
HW_CLOUD_RECOMMENDED = "CLOUD — RECOMMENDED"
HW_NOT_RUNNABLE = "NOT RUNNABLE"

# Adapter Health Check Statuses
STATUS_READY = "READY"
STATUS_MEMORY_LIMITED = "MEMORY_LIMITED"
STATUS_DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
STATUS_WEIGHTS_MISSING = "WEIGHTS_MISSING"
STATUS_CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
STATUS_CLOUD_RECOMMENDED = "CLOUD_RECOMMENDED"
STATUS_INCOMPATIBLE = "INCOMPATIBLE"

# VRAM Thresholds for RTX 3050 6GB Laptop GPU
VRAM_MAX_TOTAL_GB = 6.0
VRAM_SAFE_CEILING_GB = 4.8  # Leave ~1.2GB for Windows OS & Display Desktop Window Manager
