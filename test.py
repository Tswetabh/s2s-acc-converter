#!/usr/bin/env python3
"""Faster-Whisper STT Test Entry Point.

Loads the local CT2 model from weights/stt/faster-whisper-large-v3-turbo-ct2.
Delegates to scripts/stt/transcribe.py or runs directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_STT = ROOT / "scripts" / "stt"
if str(SCRIPTS_STT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_STT))

from transcribe import main

if __name__ == "__main__":
    sys.exit(main())