# Model Architecture & Feature Matrix

This matrix provides a detailed technical comparison of the 14 Text-to-Speech models evaluated in the Unified TTS POC platform.

---

| # | Model | Parameters | Official Version / Revision | Repo & Architecture | Supported Languages | Voice Cloning | Streaming Support | Code / Weights License | Hardware Profile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Kokoro v1.0** | 82M | v1.0 | [hexgrad/kokoro](https://github.com/hexgrad/kokoro)<br>StyleTTS 2 + ISTFTNet | en (US/GB), hi, fr, ja, it, es, pt, zh | Predefined voices (40+ styles) | Experimental (chunked) | Apache 2.0 / Apache 2.0 | ✅ LOCAL — COMFORTABLE (~1.2 GB VRAM) |
| 2 | **StyleTTS 2** | ~150M | v2.0 | [yingwp/StyleTTS2](https://github.com/yingwp/StyleTTS2)<br>Style Diffusion + ISTFT | en | Voice adaptation / reference style | Not supported (autoregressive style) | MIT / Academic | ⚠️ LOCAL — MEMORY LIMITED (~2.8 GB VRAM) |
| 3 | **CosyVoice 2** | 0.5B | CosyVoice2-0.5B | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)<br>Qwen2.5-0.5B + Flow Matching | zh, en, ja, ko, yue, dialects | Zero-shot cloning (3s prompt) | Native chunked (150ms TTFA) | Apache 2.0 / Apache 2.0 | ⚠️ LOCAL — MEMORY LIMITED (~4.2 GB VRAM) |
| 4 | **CosyVoice 3** | 0.5B | Fun-CosyVoice3-0.5B-2512 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)<br>RL-guided Flow Matching | 9 languages, 18 dialects | High-fidelity zero-shot cloning | Native chunked (150ms TTFA) | Apache 2.0 / Apache 2.0 | ⚠️ LOCAL — MEMORY LIMITED (~4.5 GB VRAM) |
| 5 | **F5-TTS** | ~350M | F5-TTS Base | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)<br>Flow Matching ConvNeXt | en, zh | Zero-shot cloning (reference audio) | Chunked / Experimental | MIT / CC-BY-NC 4.0 | ⚠️ LOCAL — MEMORY LIMITED (~3.8 GB VRAM) |
| 6 | **Piper** | 15–50M | v1.2.0 | [rhasspy/piper](https://github.com/rhasspy/piper)<br>VITS / ONNX Runtime | 40+ languages (individual models) | Predefined single/multi-speaker | Chunked ONNX streaming | MIT / MIT & CC-BY | ✅ LOCAL — COMFORTABLE (CPU/GPU) (<0.2 GB VRAM) |
| 7 | **Qwen3-TTS** | 0.6B | Qwen3-TTS-12Hz-0.6B-Base | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)<br>Qwen LLM + 12Hz Codec | 10 languages (en, zh, ja, ko, de, fr, ru, pt, es, it) | Zero-shot cloning (3s prompt) | Native streaming (~97ms TTFA) | Apache 2.0 / Apache 2.0 | ✅ LOCAL — COMFORTABLE (~3.4 GB VRAM) |
| 8 | **Chatterbox-Turbo** | 350M | Turbo v1 | [ResembleAI/chatterbox](https://github.com/resemble-ai/chatterbox)<br>1-Step Diffusion | en | Zero-shot voice cloning | Native low-latency (1-step) | MIT / MIT | ⚠️ LOCAL — MEMORY LIMITED (~3.5 GB VRAM) |
| 9 | **XTTS-v2** | 460M | v2.0.2 | [coqui-ai/TTS](https://github.com/coqui-ai/TTS)<br>GPT + HiFi-GAN Vocoder | 17 languages | Zero-shot cloning (6s reference) | Native chunked streaming | CPML / CPML (Non-Commercial) | ⚠️ LOCAL — MEMORY LIMITED (~4.2 GB VRAM) |
| 10 | **Orpheus** | 3B | orpheus-3b-0.1-ft | [CanopyAI/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)<br>Llama-3B Autoregressive | en | Zero-shot voice cloning | Streaming supported (~150ms TTFA) | Apache 2.0 / Llama 3 Community | ☁️ CLOUD — RECOMMENDED (>7.0 GB VRAM) |
| 11 | **MegaTTS3** | ~450M | MegaTTS3 Base | [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)<br>Sparse Latent Diffusion Transformer | en, zh | Zero-shot cloning (WavVAE) | Experimental | Research / CC-BY-NC | ☁️ CLOUD — RECOMMENDED (>6.5 GB VRAM) |
| 12 | **Dia2** | 1B–2B | Dia2 1B / 2B | [nari-labs/dia2](https://github.com/nari-labs/dia2)<br>Autoregressive Dialogue LLM | en | Zero-shot speaker prefixing | Native streaming dialogue | Apache 2.0 / Research | ☁️ CLOUD — RECOMMENDED (>8.0 GB VRAM) |
| 13 | **Microsoft VibeVoice** | 1.5B | VibeVoice-1.5B | [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)<br>Qwen2.5-1.5B + 7.5Hz Diffusion | en, multilingual (50+ ASR, core TTS) | Multi-speaker conversational | Long-form conversational streaming | MIT / Microsoft Research | ☁️ CLOUD — RECOMMENDED (>9.0 GB VRAM) |
| 14 | **Fish Audio S2 Pro** | Dual-AR | S2 Pro | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)<br>Dual-Autoregressive Transformer | 80+ languages | Zero-shot cloning & prompt conditioning | SGLang Native Streaming | Apache 2.0 / Fish Audio Research | ☁️ CLOUD — RECOMMENDED (>12.0 GB VRAM) |

---

## 2. Voice Cloning Taxonomy

The evaluation distinguishes four strictly separate voice customization paradigms:

1. **Predefined Voice**: Selecting from a curated bank of fixed speaker embeddings (e.g., Kokoro voice tags `af_bella`, `am_adam`, Piper models).
2. **Zero-Shot Voice Cloning**: Conditioning the model on 3–10 seconds of unseen reference audio without any parameter updates (e.g., Qwen3-TTS, CosyVoice 2/3, F5-TTS, XTTS-v2).
3. **Voice Adaptation / Prefixing**: Supplying a short acoustic prefix as a prompt to align speaker timbre and conversational tempo (e.g., Dia2, StyleTTS 2 reference style).
4. **Fine-Tuning**: Gradient descent updates performed on speaker-specific datasets (e.g., full checkpoint adaptation).
