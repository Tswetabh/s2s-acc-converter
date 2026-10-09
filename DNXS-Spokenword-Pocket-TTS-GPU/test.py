#!/usr/bin/env python3
"""Pocket TTS - Voice Cloning Test (Root Compatibility Wrapper).

The primary TTS test script is now located at:
    scripts/tts/test.py

This wrapper preserves backwards compatibility for existing workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SCRIPTS = ROOT.parent / "scripts" / "tts"
SCRIPTS_DIR = PARENT_SCRIPTS if PARENT_SCRIPTS.exists() else (ROOT / "scripts" / "tts")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import test as tts_test
    if __name__ == "__main__":
        sys.exit(tts_test.main())
except ImportError:
    import subprocess
    cmd = [sys.executable, str(SCRIPTS_DIR / "test.py")] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)