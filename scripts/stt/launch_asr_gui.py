#!/usr/bin/env python3
"""Launcher for the standalone ASR GUI.

Wraps ASR/asr_gui.py with repo-relative paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
ASR_DIR = ENGINE_DIR / "ASR"

if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    os.chdir(str(ENGINE_DIR))
    import asr_gui
    if hasattr(asr_gui, "main"):
        sys.exit(asr_gui.main())
