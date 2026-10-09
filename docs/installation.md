# Installation & Setup Guide

This guide details how to install and configure the Unified Open-Source TTS Evaluation Platform on Windows 11 and Linux, with explicit attention to hardware memory constraints and isolated secondary drive configuration.

---

## 1. System Requirements

- **Operating System**: Windows 11 (64-bit) or Ubuntu 22.04+ LTS
- **Python**: 3.12 (Recommended for active compatibility with PyTorch 2.6 and scientific packages)
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6.0 GB VRAM) or higher with CUDA 12.4+ drivers
- **RAM**: Minimum 16 GB System Memory
- **Disk Space**: At least 30 GB free on secondary drive (`D:` drive) for model weights and audio outputs.

---

## 2. Windows 11 Quickstart

### Step 1: Clone Repository & Create Virtual Environment on D: Drive
Open PowerShell:
```powershell
# Navigate to project location
cd D:\tts-poc

# Create Python 3.12 virtual environment
py -3.12 -m venv .venv
```

### Step 2: Configure Cache Paths to Preserve Drive C
Set system cache environment variables to point to Drive D:
```powershell
$env:HF_HOME = "D:\tts-poc-cache\huggingface"
$env:TORCH_HOME = "D:\tts-poc-cache\torch"
$env:PIP_CACHE_DIR = "D:\tts-poc-cache\pip"
```

### Step 3: Install Core Dependencies & PyTorch (CUDA 12.4)
```powershell
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --cache-dir D:\tts-poc-cache\pip

# Install Base Requirements
pip install -r requirements\base.txt --cache-dir D:\tts-poc-cache\pip
```

### Step 4: Install Model-Specific Adapters
Each model's dependencies are cleanly isolated in `requirements/`:
```powershell
# For Kokoro v1.0 82M:
pip install kokoro>=0.8.4 soundfile misaki --cache-dir D:\tts-poc-cache\pip

# For Piper (ultra-low latency CPU):
pip install piper-tts onnxruntime --cache-dir D:\tts-poc-cache\pip

# For Qwen3-TTS 0.6B:
pip install qwen-tts transformers accelerate --cache-dir D:\tts-poc-cache\pip
```

---

## 3. Launching the Web UI

Run the multi-page Streamlit application:
```powershell
streamlit run app\main.py
```
This opens the unified evaluation suite in your browser (`http://localhost:8501`), featuring:
1. **System Diagnostics**: Real-time VRAM, RAM, and CUDA hardware detection.
2. **Model Catalog**: Complete specifications, licenses, and local status.
3. **Interactive Audio Generation**: Synthesis with waveform viewer and metrics.
4. **Automated Benchmark Runner**: Batch test suite execution.
5. **Comparison Dashboard**: RTF, VRAM, and quality vs efficiency charts.
6. **Human Blind Evaluation**: Double-blind A/B/C testing.

---

## 4. CLI Execution Commands

Check hardware diagnostics:
```powershell
python scripts\system_info.py
```

Run single model synthesis:
```powershell
python scripts\run_model.py --model kokoro --text "Artificial intelligence is changing the way we speak."
```

Benchmark a specific model:
```powershell
python scripts\benchmark_model.py --model kokoro
```

Run batch benchmark (skipping cloud-only models):
```powershell
python scripts\benchmark_all.py --local-only
```
