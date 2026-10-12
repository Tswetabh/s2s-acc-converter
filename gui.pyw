#!/usr/bin/env pythonw
"""Windowless desktop launcher for S2S Voice & Accent Studio GUI."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET_PYTHONW = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "pythonw.exe"
TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"
EXE = TARGET_PYTHONW if TARGET_PYTHONW.exists() else TARGET_PYTHON

if EXE.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(EXE.resolve()))
    if curr_exe != target_exe:
        subprocess.Popen([str(EXE), str(ROOT / "scripts" / "gui_app.py")] + sys.argv[1:])
        sys.exit(0)

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui_app import main

if __name__ == "__main__":
    main()
