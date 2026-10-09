# Cloud GPU Execution Guide

This runbook provides complete recipes for running heavy / cloud-recommended TTS models on remote GPU environments (Google Colab, Kaggle, RunPod, Vast.ai, and Lambda Labs) when local VRAM (RTX 3050 6 GB) is insufficient.

---

## 1. Supported Cloud Environments

| Provider | Recommended GPU Tier | Typical VRAM | Approximate Cost | Best Suited Models |
|---|---|---|---|---|
| **Google Colab (Pro)** | A100 (40/80 GB) or L4 (24 GB) | 24–40 GB | ~$0.20/hr compute credits | Orpheus 3B, VibeVoice 1.5B, Fish S2 Pro |
| **Kaggle** | 2x T4 (16 GB each) or P100 (16 GB) | 16 GB | Free (30 hrs/wk) | Dia2 1B, MegaTTS3, CosyVoice 3 |
| **RunPod.io** | RTX 4090 / A10G / L4 / A100 | 24–80 GB | $0.44–$1.89/hr | All models at full precision / high batch |
| **Vast.ai** | RTX 3090 / RTX 4090 (24 GB) | 24 GB | $0.25–$0.40/hr | Orpheus 3B, Dia2 2B, MegaTTS3 |

---

## 2. Cloud Environment Setup (One-Click Bootstrap)

On any Linux cloud instance (Ubuntu 22.04 + CUDA 12.x), clone and run the unified repository:

```bash
# 1. Clone repository
git clone https://github.com/your-org/tts-poc.git
cd tts-poc

# 2. Create isolated virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements/base.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 3. Model-Specific Cloud Recipes

### 1. Orpheus (Canopy Labs 3B)
- **Recommended GPU**: A10G (24 GB) or A100 (40 GB)
- **Minimum VRAM**: 12 GB (in 4-bit) / 18 GB (in FP16)
- **Setup Commands**:
  ```bash
  pip install -r requirements/orpheus.txt
  pip install transformers accelerate bitsandbytes
  ```
- **Execution**:
  ```bash
  python scripts/run_model.py --model orpheus --text "I can't believe we actually did it! <laugh> This is amazing!" --device cuda
  python scripts/benchmark_model.py --model orpheus
  ```

### 2. Microsoft VibeVoice 1.5B
- **Recommended GPU**: A10G / L4 (24 GB)
- **Setup Commands**:
  ```bash
  pip install -r requirements/vibevoice.txt
  ```
- **Long-Form Podcast Benchmark**:
  ```bash
  python scripts/run_model.py --model vibevoice --text "Welcome to episode 42 of the AI research podcast. Today we delve into acoustic tokenization." --device cuda
  ```

### 3. Fish Audio S2 Pro
- **Recommended GPU**: A100 (80 GB) or RTX 4090 (24 GB)
- **Setup Commands**:
  ```bash
  pip install -r requirements/fishs2.txt
  pip install sglang
  ```
- **Inference & Streaming Benchmark**:
  ```bash
  python scripts/run_model.py --model fishs2 --text "Zero-shot voice cloning with dual autoregressive semantic transformers." --device cuda
  ```

### 4. Nari Labs Dia2 (1B / 2B)
- **Recommended GPU**: RTX 3090 / 4090 / A10G (24 GB)
- **Setup Commands**:
  ```bash
  pip install -r requirements/dia2.txt
  ```
- **Multi-Speaker Dialogue Execution**:
  ```bash
  python scripts/run_model.py --model dia2 --text "[S1] Hey, did you finish reviewing the TTS benchmark? [S2] Yes, Kokoro and Qwen are performing exceptionally well!" --device cuda
  ```

### 5. ByteDance MegaTTS3
- **Recommended GPU**: T4 (16 GB) / A10G (24 GB)
- **Setup Commands**:
  ```bash
  pip install -r requirements/megatts3.txt
  ```
- **Benchmark Command**:
  ```bash
  python scripts/benchmark_model.py --model megatts3
  ```

---

## 4. Exporting Results from Cloud to Local

After completing benchmarks on a cloud instance, export the generated audio files and updated metrics:

```bash
# Package audio outputs and benchmark CSV
tar -czvf cloud_results.tar.gz outputs/benchmarks/results.csv outputs/audio/

# Download to your local machine via SCP or Colab files download
# (From local terminal):
scp user@cloud-ip:~/tts-poc/cloud_results.tar.gz D:/tts-poc/
tar -xzvf D:/tts-poc/cloud_results.tar.gz -C D:/tts-poc/
```
The local Streamlit comparison dashboard will immediately incorporate the cloud benchmark numbers alongside local measurements.
