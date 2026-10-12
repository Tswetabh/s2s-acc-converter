#!/usr/bin/env python3
"""Launcher for S2S Voice & Accent Studio GUI.

Auto-redirects to the dedicated virtual environment if executed with global/other Python.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"

if TARGET_PYTHON.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(TARGET_PYTHON.resolve()))
    if curr_exe != target_exe:
        result = subprocess.run([str(TARGET_PYTHON), str(ROOT / "scripts" / "gui_app.py")] + sys.argv[1:], check=False)
        sys.exit(result.returncode)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui_app import main

if __name__ == "__main__":
    main()
