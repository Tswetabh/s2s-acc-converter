#!/usr/bin/env python3
"""Root entry point for S2S Pipeline.

Converts M4A/audio -> Faster-Whisper STT -> Pocket TTS Voice Clone.
Delegates to scripts/pipeline.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import main

if __name__ == "__main__":
    sys.exit(main())
