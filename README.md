# Unified Open-Source TTS POC Evaluation Platform

A unified, evidence-based Text-to-Speech (TTS) evaluation and benchmarking platform designed to fairly compare 14 open-source / open-weight models under consumer hardware constraints.

---

## 1. Test Workstation Profile

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM)
- **CPU**: AMD Ryzen 7 7840HS (8 cores / 16 threads)
- **System Memory**: 16.0 GB RAM
- **Operating System**: Windows 11 (64-bit)
- **CUDA Acceleration**: CUDA 12.4 / PyTorch 2.6.0+cu124
- **Physical Storage**: Secondary Drive (`D:\tts-poc` and `D:\tts-poc-cache`) — zero storage impact on Drive C.

---

## 2. Evaluated Models (All 14 Adapters)

| # | Model | Parameters | Languages | Hardware Profile | Code License | Weights License | Key Capability |
|---|---|---|---|---|---|---|---|
| 1 | **Kokoro v1.0** | 82M | en, hi, es, fr, ja, it, pt, zh | ✅ LOCAL — COMFORTABLE | Apache 2.0 | Apache 2.0 | High naturalness, fast real-time (RTF <0.1–0.5) |
| 2 | **Piper** | 15–50M | 40+ languages | ✅ LOCAL — COMFORTABLE (CPU) | MIT | MIT / CC-BY | Ultra-low CPU latency (<200MB RAM) |
| 3 | **Qwen3-TTS** | 0.6B | 10 languages | ✅ LOCAL — COMFORTABLE | Apache 2.0 | Apache 2.0 | 3s zero-shot cloning, streaming (~97ms TTFA) |
| 4 | **StyleTTS 2** | ~150M | en | ⚠️ LOCAL — MEMORY LIMITED | MIT | Academic | Style diffusion, prosody expressiveness |
| 5 | **Chatterbox-Turbo** | 350M | en | ⚠️ LOCAL — MEMORY LIMITED | MIT | MIT | 1-step diffusion, paralinguistic tags (`[laugh]`) |
| 6 | **CosyVoice 2** | 0.5B | zh, en, ja, ko, dialects | ⚠️ LOCAL — MEMORY LIMITED | Apache 2.0 | Apache 2.0 | Flow matching zero-shot cloning, chunked stream |
| 7 | **CosyVoice 3** | 0.5B | 9 langs, 18 dialects | ⚠️ LOCAL — MEMORY LIMITED | Apache 2.0 | Apache 2.0 | RL-optimized flow matching, robust prosody |
| 8 | **F5-TTS** | ~350M | en, zh | ⚠️ LOCAL — MEMORY LIMITED | MIT | CC-BY-NC 4.0 | Flow matching zero-shot cloning |
| 9 | **XTTS-v2** | 460M | 17 languages | ⚠️ LOCAL — MEMORY LIMITED | CPML | CPML (Non-Commercial) | Cross-lingual zero-shot voice cloning |
| 10 | **Orpheus** | 3B | en | ☁️ CLOUD — RECOMMENDED | Apache 2.0 | Llama 3 Community | Empathetic speech, `<laugh>` tags, Llama-3B |
| 11 | **MegaTTS3** | ~450M | en, zh | ☁️ CLOUD — RECOMMENDED | Research | CC-BY-NC | Sparse latent diffusion transformer, WavVAE |
| 12 | **Dia2** | 1B–2B | en | ☁️ CLOUD — RECOMMENDED | Apache 2.0 | Research Preview | Multi-speaker dialogue (`[S1]`, `[S2]`) |
| 13 | **Microsoft VibeVoice** | 1.5B | en, multilingual | ☁️ CLOUD — RECOMMENDED | MIT | Microsoft Research | 90-min conversational podcast synthesis |
| 14 | **Fish Audio S2 Pro** | Dual-AR | 80+ languages | ☁️ CLOUD — RECOMMENDED | Apache 2.0 | Fish Research License | Dual-Autoregressive high-end quality reference |

---

## 3. Architecture & Project Layout

```text
tts-poc/
├── README.md
├── models/                     # Model-specific integration adapters & loaders
│   ├── tts/
│   │   └── pocket_tts/         # Pocket TTS adapter
│   ├── stt/
│   │   ├── faster_whisper/     # Faster-Whisper CT2 integration adapter
│   │   └── whisper_cpp/        # Whisper.cpp backend adapter
│   └── other/
├── weights/                    # Actual downloaded/trained model weights (Git-ignored)
│   ├── stt/
│   │   └── faster-whisper-large-v3-turbo-ct2/  # Local CT2 model files (1.61 GB)
│   ├── tts/
│   └── other/
├── scripts/                    # CLI execution & evaluation scripts
│   ├── tts/                    # TTS executable scripts (test.py, launch_gui.py, convert.py)
│   ├── stt/                    # STT executable scripts (transcribe.py, launch_asr_gui.py)
│   ├── benchmark_all.py        # Benchmark suite runner
│   ├── benchmark_model.py
│   ├── clone_voice.py          # Zero-shot voice cloning CLI
│   ├── run_model.py
│   └── system_info.py
├── Testinput/                  # Test audio inputs and reference voices
│   ├── Swetabh_Input.wav
│   ├── Swetabh Input.m4a
│   ├── arctic_a0005.wav
│   └── test.mp3
├── Output/                     # Generated audio outputs
│   └── cloned_output.wav
├── DNXS-Spokenword-Pocket-TTS-GPU/  # Dedicated Pocket TTS & ASR engine repository
│   ├── pocket_tts/             # Core Pocket TTS Python package
│   ├── ASR/                    # ASR verification & alignment subsystem
│   ├── tools/                  # Developer utilities & CUDA graph probes
│   └── tests/                  # Automated test suite
├── app/                        # Multi-page Streamlit comparative evaluation dashboard
│   ├── main.py
│   ├── models/                 # Evaluated model adapters
│   └── ui/                     # Interactive Streamlit pages
├── data/                       # Standardized test corpus & reference audio
├── outputs/                    # Telemetry, benchmarks & human evaluation records
├── requirements/               # Model-isolated dependency files
└── docs/                       # Technical documentation
```

---

## 4. Quickstart & Usage

### 1. Launch Interactive Streamlit UI
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app\main.py
```
Access the application at `http://localhost:8501`.

### 2. Run CLI Hardware Diagnostics
```powershell
python scripts\system_info.py
```

### 3. Synthesize Speech with a Model
```powershell
python scripts\run_model.py --model kokoro --text "Artificial intelligence is changing the way people learn and work."
```

### 4. Run Standardized Benchmark on a Model
```powershell
python scripts\benchmark_model.py --model kokoro
```

### 5. Run Batch Benchmark (Local Models Only)
```powershell
python scripts\benchmark_all.py --local-only
```

### 6. Run Zero-Shot Voice Cloning
Clone any voice in real time using 3-10s reference audio:
```powershell
# Using the Hindi Tyagi voice preset:
python scripts\clone_voice.py --ref-audio tyagi --text "नमस्ते, यह मेरा नया क्लोन किया हुआ आवाज़ है।"

# Or with your own custom MP3/WAV file:
python scripts\clone_voice.py --ref-audio "C:\Users\tripa\Downloads\Hindi_F_Tyagi.mp3" --text "Hello world! This is a cloned voice."

# Interactive mode (prompts for inputs):
python scripts\clone_voice.py
```

### 7. Run Speech-to-Text (STT) Transcription
Transcribe speech using Faster-Whisper with your choice of model:
```powershell
# Using Tiny Whisper (ultra-fast, lightweight, ~75MB):
python scripts\stt\transcribe.py --model tiny --audio Testinput\Swetabh_Input.wav

# Or using the root shortcut:
python test.py --model tiny --audio Testinput\Swetabh_Input.wav

# Using Large-v3-Turbo (high accuracy, ~1.61GB):
python scripts\stt\transcribe.py --model turbo --audio Testinput\Swetabh_Input.wav
```

### 8. Run Unified Speech-to-Speech (M4A -> STT -> TTS) Pipeline
Convert laptop `.m4a` recordings, transcribe with Faster-Whisper, and synthesize with Pocket TTS voice cloning:
```powershell
# 1. Automatically grab the newest recording from Windows Sound Recordings:
python pipeline.py

# 2. Interactively pick from available recordings in Sound Recordings folder:
python pipeline.py --select

# 3. Transcribe an explicit M4A file with high accuracy turbo model:
python pipeline.py --audio "Testinput\Swetabh Input.m4a" --stt-model turbo

# 4. Use custom reference voice (defaults to 'Testinput\Anoop voice line.wav'):
python pipeline.py --reference "Testinput\Anoop voice line.wav"

# 5. Specify custom recordings directory or output file:
python pipeline.py --recordings-dir "C:\Users\tripa\Documents\Sound Recordings" --output "Output\my_result.wav"
```

### 9. Compile Final Markdown Evaluation Report
```powershell
python scripts\generate_report.py
```



---

## 5. Core System Invariants

1. **Strict Single-Model Residency**: Only one model is kept in VRAM at any time. Models are immediately evicted, followed by garbage collection and `torch.cuda.empty_cache()`.
2. **Zero Fabrication**: Only empirical measurements or explicitly published author numbers are recorded.
3. **Storage Isolation**: Python virtual environment (`.venv`), Pip cache (`PIP_CACHE_DIR`), Hugging Face cache (`HF_HOME`), and model weights live strictly on Drive D.
