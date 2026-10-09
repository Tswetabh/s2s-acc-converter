#!/usr/bin/env python3
"""
Ensure a runnable test environment for GPU optimization tests.

This script checks for required packages and installs them into the
current interpreter if missing. It is intentionally conservative and
only touches the test environment, never production code.

Usage:
    python tests/ensure_test_env.py
"""
import subprocess
import sys
from pathlib import Path

REQUIRED = ["beartype", "torch", "scipy"]


def main():
    """Main function to check and install missing packages."""
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg.split("[")[0])
        except ImportError:
            missing.append(pkg)

    if not missing:
        print("All required packages present.")
        return 0

    print(f"Missing packages: {missing}")
    print("Attempting pip install into current environment...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + missing
    subprocess.check_call(cmd)
    print("Install complete. Re-run your test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
