#!/usr/bin/env python3
"""Root entry point for S2S Pipeline.

Converts M4A/audio -> Faster-Whisper STT -> Pocket TTS Voice Clone.
Auto-redirects to the dedicated virtual environment if executed with global/other Python.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"

# Auto-redirect to dedicated virtual environment if not already active
if TARGET_PYTHON.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(TARGET_PYTHON.resolve()))
    if curr_exe != target_exe:
        result = subprocess.run([str(TARGET_PYTHON)] + sys.argv, check=False)
        sys.exit(result.returncode)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import main

if __name__ == "__main__":
    sys.exit(main())
