# Hardware Compatibility Matrix

This document provides empirical hardware classification guidelines for running Text-to-Speech models on consumer hardware, with specific reference to our test workstation:

- **Target Workstation**: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM)
- **Host System**: AMD Ryzen 7 7840HS, 16.0 GB System RAM, Windows 11
- **VRAM Budget**: ~4.8 GB safe ceiling (~1.2 GB reserved for OS Desktop Window Manager & display buffers).

---

## 1. Classification Categories

| Classification | Definition | Target Hardware Profile |
|---|---|---|
| **✅ LOCAL — COMFORTABLE** | Executes with ample memory headroom (<2.5 GB peak VRAM). Zero memory pressure. | RTX 3050 6 GB / Modern GPU |
| **⚠️ LOCAL — MEMORY LIMITED** | Fits in 3.0–5.2 GB VRAM or requires FP16 / CPU offloading / chunking. Susceptible to OOM on long texts (>80 words). | RTX 3050 6 GB (Caution) |
| **☁️ CLOUD — RECOMMENDED** | Requires >5.5 GB VRAM or benefits strongly from 16 GB+ VRAM (A10G, T4, L4, A100). | Cloud GPU (RunPod, Colab, Vast.ai) |
| **❌ NOT RUNNABLE** | Host platform, OS binary, or Python dependency conflict prevents execution. | Incompatible platform |

---

## 2. Model Compatibility Breakdown

| # | Model | Parameters | Target Architecture | Local Classification | Cloud Recommendation | Rationale & Memory Footprint |
|---|---|---|---|---|---|---|
| 1 | **Kokoro v1.0** | 82M | StyleTTS 2 / ISTFTNet | **✅ LOCAL — COMFORTABLE** | Optional | Extremely lightweight (<1.2 GB VRAM peak). Runs fast on RTX 3050 and CPU. |
| 2 | **Piper** | ~15–50M | VITS / ONNX | **✅ LOCAL — COMFORTABLE (CPU/GPU)** | Not needed | Ultra-fast on CPU (<200 MB RAM). Perfect for low-resource edge deployment. |
| 3 | **Qwen3-TTS 0.6B** | 0.6B | Decoder LLM + 12Hz Codec | **✅ LOCAL — COMFORTABLE** | High-concurrency | Fits in ~3.2–3.8 GB VRAM in FP16/BF16. |
| 4 | **StyleTTS 2** | ~150M | Style Diffusion + ISTFT | **⚠️ LOCAL — MEMORY LIMITED** | Optional | ~2.5–3.5 GB VRAM. Requires monotonic align and phonemizer setup on Windows. |
| 5 | **Chatterbox-Turbo** | 350M | 1-Step Diffusion | **⚠️ LOCAL — MEMORY LIMITED** | High batch | ~2.8–3.8 GB VRAM. Real-time inference fits comfortably when batched lightly. |
| 6 | **CosyVoice 2** | 0.5B | Qwen2.5-0.5B + Flow Matching | **⚠️ LOCAL — MEMORY LIMITED** | Production serving | ~3.8–4.6 GB VRAM in FP16. High memory allocation during flow matching steps. |
| 7 | **CosyVoice 3** | 0.5B | Differentiable Reward FM | **⚠️ LOCAL — MEMORY LIMITED** | Production serving | ~4.0–4.8 GB VRAM. Safe for short/medium utterances; long texts may hit 6GB ceiling. |
| 8 | **F5-TTS** | ~350M | Flow Matching ConvNeXt | **⚠️ LOCAL — MEMORY LIMITED** | Faster generation | ~3.5–4.5 GB VRAM. Diffusion/flow-matching steps require sequential VRAM buffers. |
| 9 | **XTTS-v2** | 460M | GPT-style autoregressive + HiFi-GAN | **⚠️ LOCAL — MEMORY LIMITED** | Multi-speaker scale | ~3.6–4.5 GB VRAM in FP16. Known to cause CUDA OOM on long texts without chunking. |
| 10 | **Orpheus** | 3B | Llama-3B backbone | **☁️ CLOUD — RECOMMENDED** | A10G / L4 (24GB) | 3B parameters require ~6.8 GB VRAM in FP16 and >4.2 GB in 4-bit quantized. RTX 3050 6GB lacks headroom. |
| 11 | **MegaTTS3** | ~450M | Sparse Latent Diffusion Transformer | **☁️ CLOUD — RECOMMENDED** | T4 / A10G | Latent diffusion transformer + WavVAE encoder requires substantial continuous memory pools. |
| 12 | **Dia2** | 1B–2B | Multi-speaker Dialogue LLM | **☁️ CLOUD — RECOMMENDED** | A10G (24GB) | 1B/2B dialogue models require 5.5–9.0 GB VRAM for dual-speaker context buffers. |
| 13 | **Microsoft VibeVoice** | 1.5B | Qwen2.5-1.5B + 7.5Hz Tokenizer | **☁️ CLOUD — RECOMMENDED** | A10G / L4 | Designed for 90-minute conversational synthesis; context memory exceeds 6 GB. |
| 14 | **Fish Audio S2 Pro** | Dual-AR | Dual-Autoregressive Transformer | **☁️ CLOUD — RECOMMENDED** | A100 / L4 / H100 | S2 Pro requires >8–12 GB VRAM for full precision Dual-AR sampling. High-end quality reference. |

---

## 3. Safe Execution Protocol for 6 GB VRAM

1. **Strict Single-Model Lifecycle**: Only one model is loaded at any time.
2. **Explicit Cache Reclamation**: Run `torch.cuda.empty_cache()` and `gc.collect()` before and after every inference call.
3. **Inference Context**: Enforce `with torch.inference_mode():` to disable computation graph caching.
4. **Long-Form Protection**: Texts exceeding 100 words should be chunked by sentences for models in the `⚠️ LOCAL — MEMORY LIMITED` category.
