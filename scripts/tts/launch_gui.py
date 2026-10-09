#!/usr/bin/env python3
"""Audiobook Generator GUI Launcher.

Entry point for launching the Pocket TTS desktop GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repository root and engine dir to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT_DIR / "DNXS-Spokenword-Pocket-TTS-GPU"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure working directory is the engine directory
os.chdir(str(ENGINE_DIR))

from pocket_tts.gui.main_window import main

if __name__ == "__main__":
    main()
