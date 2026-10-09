# Standardized Benchmark Methodology

This document outlines the evaluation framework, measurement metrics, and testing procedures implemented across all 14 Text-to-Speech models.

---

## 1. Objectives & Guiding Principles

The platform is designed to answer:
> *"Among current open-source/open-weight TTS models, which model gives the best combination of naturalness, expressiveness, voice cloning capability, multilingual support, latency, and hardware efficiency for our use case?"*

### Core Evaluation Rules:
1. **Never Fabricate Numbers**: All metrics reported in the evaluation must be either:
   - Officially published by original paper authors/model creators (clearly labeled as *Officially Reported*), or
   - Directly measured on the local host or documented cloud instance (clearly labeled as *Measured*).
2. **Standardized Corpora**: All models must synthesize identical test utterances from `data/evaluation_texts.json` to ensure fair comparison.
3. **Strict Memory Isolation**: Only one model is loaded at a time. The platform measures baseline memory, peak memory, and post-unload residual memory.

---

## 2. Standardized Test Corpus (`data/evaluation_texts.json`)

To stress-test phonetic accuracy, prosody, and linguistic edge cases, the benchmark evaluates 12 categories:

1. **Normal**: Standard conversational prose with balanced phonetic distribution.
2. **Conversation**: Casual interpersonal dialogue with colloquial pacing.
3. **Question**: Interrogative sentences evaluating rising pitch contours.
4. **Numbers**: Currency, percentages, and complex numeric strings.
5. **Dates**: Full dates, ordinal markers, and clock times.
6. **Technical**: High-density scientific and technical nomenclature (transformers, attention).
7. **Expressive**: High-energy exclamations testing emotional inflection.
8. **Punctuation**: Pauses, dashes, ellipses, and quotes testing prosodic rhythm.
9. **Proper Nouns**: Phonetic transliteration and international names.
10. **Long-Form**: Multi-sentence continuous paragraph (100–150 words) testing drift, memory spikes, and hallucination degradation.
11. **Hindi**: Devanagari script evaluation for non-Latin multilingual capabilities.
12. **Hinglish**: Code-mixed Hindi-English sentences reflecting real-world bilingual usage.

If a model does not support a given language (e.g., Hindi), the system flags the test as `NOT_SUPPORTED` rather than failing or reporting skewed quality metrics.

---

## 3. Performance Metrics & Measurement Protocol

| Metric | Symbol / Unit | Calculation / Protocol |
|---|---|---|
| **Load Time** | `load_time_sec` (s) | Monotonic wall time from load initiation until model weights and vocoders are resident in VRAM and ready for inference. |
| **Generation Time** | `gen_time_sec` (s) | Monotonic wall time from synthesis request to WAV buffer completion. |
| **Audio Duration** | `audio_dur_sec` (s) | Total duration of generated PCM audio stream (`frames / sample_rate`). |
| **Real-Time Factor** | `RTF` | `generation_time_sec / audio_duration_sec`. (Values < 1.0 indicate faster-than-real-time generation). |
| **Time-to-First-Audio** | `TTFA` (s / ms) | Applicable strictly for streaming-capable models; wall time from request to emission of the first playable audio chunk. Recorded as `N/A` for batch-only models. |
| **Peak VRAM** | `vram_peak_gb` (GB) | Maximum GPU memory allocated during synthesis via `torch.cuda.max_memory_allocated(0)` + baseline device usage. |
| **RAM Utilization** | `ram_peak_gb` (GB) | Peak system memory sampled via `psutil.virtual_memory()`. |

---

## 4. Subjective Quality Evaluation

Automated metrics cannot fully capture prosodic naturalness. The platform includes a dedicated human evaluation module:

- **Scoring Range**: 1 (Poor) to 5 (Flawless) across:
  - Naturalness
  - Pronunciation
  - Prosody & Rhythm
  - Expressiveness
  - Speaker Similarity (voice cloning tests only)
  - Overall Quality
- **Blind Evaluation Mode**: Model identities are masked (`Sample A`, `Sample B`, `Sample C`) to eliminate experimenter bias.
- **Configurable Weighted Scoring**: Customizable weighting across quality and performance parameters.
