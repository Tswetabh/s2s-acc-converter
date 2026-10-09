# TTS Evaluation Platform — Licensing & Legal Matrix

This document details the code license, model weights license, training-data restrictions, commercial usability, attribution requirements, and other limitations for all 14 evaluated Text-to-Speech models.

> **CRITICAL LEGAL DISTINCTION**: In modern generative AI, "open source code" and "open weights" are frequently governed by separate, distinct licenses. An MIT-licensed inference wrapper does NOT grant commercial rights if the underlying model checkpoint is released under Non-Commercial (e.g., CC-BY-NC) or proprietary terms.

---

## 1. Summary Matrix

| # | Model | Code License | Weights License | Commercial Usable? | Key License Restrictions & Attribution |
|---|---|---|---|---|---|
| 1 | **Kokoro v1.0 (82M)** | Apache 2.0 | Apache 2.0 | **Yes** | Permissive; retain copyright notices. |
| 2 | **StyleTTS 2** | MIT | MIT / Academic | **Restricted** | Pretrained LibriTTS weights are for research; custom commercial training required for products. |
| 3 | **CosyVoice 2 (0.5B)** | Apache 2.0 | Apache 2.0 | **Yes** | FunAudioLLM permissive license; retain attribution. |
| 4 | **CosyVoice 3 (0.5B)** | Apache 2.0 | Apache 2.0 | **Yes** | Alibaba FunAudioLLM Apache 2.0 terms. |
| 5 | **F5-TTS** | MIT | CC-BY-NC 4.0 | **No (Weights NC)** | Source code is MIT, but official model checkpoints are explicitly Non-Commercial (CC-BY-NC 4.0). Commercial deployment requires retraining from scratch. |
| 6 | **Piper** | MIT | MIT / CC-BY-4.0 | **Yes** | Dependent on individual voice model selected; standard English voices are MIT/CC-BY. |
| 7 | **Qwen3-TTS (0.6B)** | Apache 2.0 | Apache 2.0 | **Yes** | Alibaba Qwen open model terms; commercially permissible. |
| 8 | **Chatterbox-Turbo (350M)** | MIT | MIT | **Yes** | Resemble AI open release under MIT license. |
| 9 | **XTTS-v2** | Coqui Public Model License (CPML) | CPML Non-Commercial | **No** | CPML strictly prohibits commercial exploitation without enterprise license. |
| 10 | **Orpheus (3B)** | Apache 2.0 | Llama 3 Community | **Yes (with caveats)** | Built on Llama-3B backbone; governed by Meta Llama 3 Community License (<700M monthly active users). |
| 11 | **MegaTTS3** | Research / Apache 2.0 | CC-BY-NC / ByteDance Research | **No** | ByteDance research weights restricted to non-commercial academic evaluation. |
| 12 | **Dia2** | Apache 2.0 | Nari Labs Research | **Research Only** | Research preview weights; commercial agreement required for hosted services. |
| 13 | **Microsoft VibeVoice 1.5B** | MIT | Microsoft Research License | **No** | Released under Microsoft Research Open License; governed by Responsible AI terms, strictly non-commercial. |
| 14 | **Fish Audio S2 Pro** | Apache 2.0 | Fish Audio Research License | **No (Commercial Requires Paid Plan)** | Free for academic, personal, and research evaluation; commercial deployment requires commercial agreement with Fish Audio. |

---

## 2. In-Depth Model Breakdown

### 1. Kokoro v1.0 — 82M
- **Official Repository**: [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
- **Hugging Face**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **Code License**: Apache 2.0
- **Weights License**: Apache 2.0
- **Training Data**: Open-domain multi-speaker datasets.
- **Commercial Use**: Permitted without royalty fees.
- **Attribution**: Include Apache 2.0 notice in software distributions.

### 2. StyleTTS 2
- **Official Repository**: [yingwp/StyleTTS2](https://github.com/yingwp/StyleTTS2)
- **Code License**: MIT
- **Weights License**: Academic research terms.
- **Training Data**: LibriTTS (CC-BY-4.0) and VCTK.
- **Commercial Use**: Code can be integrated commercially, but official pretrained weights carry academic restrictions. Commercial users must train checkpoints on cleared datasets.

### 3. CosyVoice 2 — 0.5B
- **Official Repository**: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- **Hugging Face**: [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
- **Code License**: Apache 2.0
- **Weights License**: Apache 2.0
- **Commercial Use**: Permitted under standard Apache 2.0 conditions.

### 4. CosyVoice 3 (Fun-CosyVoice3-0.5B-2512)
- **Official Repository**: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- **Hugging Face / ModelScope**: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- **Code License**: Apache 2.0
- **Weights License**: Apache 2.0
- **Commercial Use**: Permitted.

### 5. F5-TTS
- **Official Repository**: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Hugging Face**: [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS)
- **Code License**: MIT
- **Weights License**: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0).
- **Commercial Use**: **STRICTLY PROHIBITED** for pretrained weights. If commercial use is required, an organization must retrain the F5-TTS model architecture independently using commercial-grade datasets.

### 6. Piper
- **Official Repository**: [rhasspy/piper](https://github.com/rhasspy/piper)
- **Code License**: MIT
- **Weights License**: Varies by voice model (e.g. `en_US-lessac-medium` is MIT, others CC-BY).
- **Commercial Use**: Fully permitted for MIT and CC-BY voice checkpoints with proper attribution.

### 7. Qwen3-TTS (0.6B)
- **Official Repository**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- **Hugging Face**: [Qwen/Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts-696fa0aecc17b9ea599b054b)
- **Code License**: Apache 2.0
- **Weights License**: Apache 2.0
- **Commercial Use**: Permitted under Alibaba Qwen terms.

### 8. Chatterbox-Turbo
- **Official Repository**: [ResembleAI/chatterbox](https://github.com/resemble-ai/chatterbox)
- **Hugging Face**: [ResembleAI/Chatterbox-Turbo](https://huggingface.co/ResembleAI)
- **Code License**: MIT
- **Weights License**: MIT
- **Commercial Use**: Permitted.

### 9. Coqui XTTS-v2
- **Official Repository**: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- **Code License**: CPML (Coqui Public Model License)
- **Weights License**: CPML (Non-Commercial)
- **Commercial Use**: Strictly prohibited for free use. Following Coqui's shutdown, legal title is held under CPML conditions requiring explicit rights grants.

### 10. Orpheus (Canopy Labs 3B)
- **Official Repository**: [CanopyAI/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- **Hugging Face**: [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
- **Code License**: Apache 2.0
- **Weights License**: Llama 3 Community License Agreement (Meta)
- **Commercial Use**: Permitted provided monthly active users do not exceed 700 million and Meta's Acceptable Use Policy is honored.

### 11. ByteDance MegaTTS3
- **Official Repository**: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)
- **Code License**: Open Research / Apache 2.0
- **Weights License**: Non-Commercial Academic License (CC-BY-NC)
- **Commercial Use**: Prohibited.

### 12. Nari Labs Dia2
- **Official Repository**: [nari-labs/dia2](https://github.com/nari-labs/dia2)
- **Code License**: Apache 2.0
- **Weights License**: Nari Labs Research License
- **Commercial Use**: Research preview only.

### 13. Microsoft VibeVoice 1.5B
- **Official Repository**: [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- **Hugging Face**: [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft)
- **Code License**: MIT
- **Weights License**: Microsoft Research Open License / Responsible AI Agreement
- **Commercial Use**: Strictly Non-Commercial.

### 14. Fish Audio S2 Pro
- **Official Repository**: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- **Hugging Face**: [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)
- **Code License**: Apache 2.0
- **Weights License**: FISH AUDIO RESEARCH LICENSE
- **Commercial Use**: Non-commercial use is free. Commercial usage requires entering into a commercial license agreement with Fish Audio.
