"""CLI script to install dependencies for individual TTS models on Drive D."""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import PROJECT_ROOT, CACHE_DIR

MODEL_REQUIREMENTS = {
    "kokoro": ["kokoro>=0.8.4", "misaki[en]>=0.4.0", "soundfile>=0.12.1"],
    "piper": ["piper-tts>=1.2.0", "onnxruntime>=1.17.0"],
    "f5tts": ["f5-tts>=0.1.0", "vocos>=0.1.0", "cached_path>=1.6.2"],
    "styletts2": ["phonemizer>=3.2.1", "librosa>=0.10.1", "nltk>=3.8.1"],
    "qwen3tts": ["transformers>=4.40.0", "accelerate>=0.28.0"],
    "chatterbox": ["diffusers>=0.27.0"],
    "cosyvoice2": ["diffusers>=0.27.0"],
    "cosyvoice3": ["modelscope>=1.14.0"],
    "xtts": ["TTS>=0.22.0"],
}

def install_model_deps(model_id: str):
    clean_id = model_id.lower().replace("-", "").replace("_", "")
    
    if clean_id == "alllocal":
        print("[INFO] Installing dependencies for all local-compatible models...")
        for m in ["kokoro", "piper", "f5tts", "styletts2", "qwen3tts"]:
            install_model_deps(m)
        return

    packages = MODEL_REQUIREMENTS.get(clean_id)
    if not packages:
        req_file = PROJECT_ROOT / "requirements" / f"{clean_id}.txt"
        if req_file.exists():
            cmd = [
                sys.executable, "-m", "pip", "install", "-r", str(req_file),
                "--cache-dir", str(CACHE_DIR / "pip")
            ]
        else:
            print(f"[WARN] No automated pip manifest for '{model_id}'.")
            return
    else:
        cmd = [
            sys.executable, "-m", "pip", "install"
        ] + packages + [
            "--cache-dir", str(CACHE_DIR / "pip")
        ]

    print(f"\n[INSTALL] Installing dependencies for {model_id}...")
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print(f"[SUCCESS] Dependencies for {model_id} installed successfully!")
    else:
        print(f"[ERROR] Failed to install dependencies for {model_id} (exit code: {res.returncode})")

def main():
    parser = argparse.ArgumentParser(description="Install model-specific dependencies into D: drive venv")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g. piper, f5tts, kokoro, all-local)")
    args = parser.parse_args()

    install_model_deps(args.model)

if __name__ == "__main__":
    main()
