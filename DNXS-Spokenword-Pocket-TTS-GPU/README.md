# Pocket TTS GPU

**DNXS Spoken Word · Audiobook Generator**

Special thanks to gstock99 for leading the way with GPU

<p align="center">
  <img src="docs/icon.png" alt="DNXS Spoken Word Pocket-TTS GPU Edition" width="880">
</p>

Pocket TTS GPU is a desktop audiobook studio built on [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts). It turns a plain-text book into spoken audio, then optionally checks every chunk with speech recognition, retries failures, and exports M4B, MP3, or WAV with chapter marks and metadata.

The main window is titled **Audiobook Generator**. Work is organized into three tabs:

| Tab                    | Purpose                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| **Generate Audiobook** | Choose a book and voice, set generation/export/ASR options, and produce one audiobook.         |
| **Regenerate Chunks**  | Open an existing TTS folder, find or repair individual chunks, then rebuild the finished book. |
| **Batch Processing**   | Run a queue of books. Each job keeps the Main-tab settings that were active when it was added. |

A companion window, **PocketGPU ASR Validator**, can re-check already generated chunks without launching a new TTS run.

---

## Contents

1. [What the program does](#what-the-program-does)
2. [Function summary by tab](#function-summary-by-tab)
3. [Repository structure](#repository-structure)
4. [Requirements](#requirements)
5. [Install](#install)
6. [Run](#run)
7. [First launch](#first-launch)
8. [User manual — Generate Audiobook](#user-manual--generate-audiobook)
9. [User manual — Export Meta Data](#user-manual--export-meta-data)
10. [User manual — Regenerate Chunks](#user-manual--regenerate-chunks)
11. [User manual — Batch Processing](#user-manual--batch-processing)
12. [User manual — PocketGPU ASR Validator](#user-manual--pocketgpu-asr-validator)
13. [Typical workflows](#typical-workflows)
14. [Output layout](#output-layout)
15. [Text conventions](#text-conventions)
16. [Configuration](#configuration)
17. [Troubleshooting](#troubleshooting)
18. [License and credits](#license-and-credits)

---

## What the program does

Pocket TTS GPU is not a one-shot “read this file” widget. It is a production pipeline:

1. **Prepare the book.** The text is split into chunks (sentence, paragraph, or word-target). Optional emotion analysis and punctuation pauses are applied.
2. **Speak the chunks.** Kyutai Pocket TTS generates each piece on GPU (or CPU). Several workers can run in parallel.
3. **Check the speech.** Optional ASR quality control listens to each chunk and compares it with the source text. Confirmed failures can be regenerated.
4. **Export the book.** Selected formats (M4B, MP3, WAV) are built from the chunk WAVs, with optional chapterization, loudness processing, tags, and cover art.
5. **Repair later.** The Regenerate tab can rewrite one chunk, then rebuild the finished files using the live Main-tab export settings.

Built-in voices: **alba** (default), **marius**, **javert**, **jean**, **fantine**, **cosette**, **eponine**, **azelma**. Any custom WAV can be used as a cloning prompt.

---

## Function summary by tab

### Generate Audiobook

The production tab. Everything that affects a new run lives here.

| Area                     | What it does                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| **File Selection**       | Pick the `.txt` book and a built-in or custom voice.                                     |
| **Show Parameters**      | Hide or show the eight parameter columns so the Results log can grow.                    |
| **Save Configuration**   | Write the current controls to `pocket_tts/config/default_config.yaml`.                   |
| **Meta Data**            | Author, series, title, cover art, and other tags used on the next export.                |
| **TTS Core Parameters**  | Temperature, end-of-speech threshold, and tail frames.                                   |
| **Chunking Settings**    | How the book is split: sentence, paragraph, or word-target.                              |
| **Pause Durations**      | Silence inserted between sentences, paragraphs, and chapters when joining audio.         |
| **Quality Settings**     | LSD refinement steps, speed variation, and emotion detection.                            |
| **Pause Injection**      | Convert punctuation (and always honor manual `[Xs]` markers) into exact digital silence. |
| **Audiobook Export**     | Which formats to write, chapter strategy, and loudness mode.                             |
| **Performance Settings** | Device (`auto` / `cpu` / `cuda`) and TTS worker count.                                   |
| **ASR Quality Control**  | Two-stage listening, language, pass threshold, and retry policy.                         |
| **Generation Progress**  | Chunks, elapsed time, ETA, completion timings, and run buttons.                          |
| **Results**              | Live log for the current session.                                                        |

Each parameter column has a **?** button with a short in-app explanation.

### Regenerate Chunks

Surgical repair of an already generated book.

| Area                     | What it does                                                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **TTS Folder Selection** | Open `Output/<Book>/TTS` (must contain `audio_chunks/` and `text_chunks/audiobook.chunks.json`).               |
| **Search Chunks**        | Find chunks by keyword, jump to a numeric ID, or load the ASR fail report.                                     |
| **Search Results**       | Click a hit to load it into the editor.                                                                        |
| **Chunk Editor**         | Edit text, set emotion, pick voice, play original, regenerate, play new, save.                                 |
| **Audiobook Rebuild**    | Restitch the book using the **current** Main-tab export checkboxes (M4B / MP3 / WAV, chapters, normalization). |

### Batch Processing

Unattended multi-book queue.

| Area                     | What it does                                                        |
| ------------------------ | ------------------------------------------------------------------- |
| **Files to Process**     | Queue of jobs. Add a whole folder of `.txt` files, remove, reorder. |
| **Overall Progress**     | File index, current-file progress, elapsed, ETA.                    |
| **Start / Pause / Stop** | Run, pause, or cancel the queue.                                    |
| **Log**                  | Per-job messages.                                                   |

Jobs are **not** taken from this tab’s own settings. Use **Add to Batch** on Generate, or **Add Folder** here. Both snapshot the Main tab at queue time. Later Main-tab edits do not change jobs already in the list.

### PocketGPU ASR Validator (companion)

Standalone Tk window (`ASR/asr_gui.py`) for re-validating existing chunks, optionally limited to a failure JSON.

---

## Repository structure

The repository is organized to clearly separate model implementations/adapters, downloaded/trained model weights, executable entry-point scripts, core application packages, tests, tools, and test inputs:

```text
DNXS-Spokenword-Pocket-TTS-GPU/
│
├── models/                     # Model-specific integration, adapters, wrappers, and configs
│   ├── tts/
│   │   ├── pocket_tts/         # Pocket TTS adapter & integration wrappers
│   │   └── <other TTS model integrations>
│   ├── stt/
│   │   ├── faster_whisper/     # Faster-Whisper CT2 adapter & loader
│   │   ├── whisper_cpp/        # Whisper.cpp backend adapter
│   │   └── <other STT model integrations>
│   └── other/                  # Future model implementations (e.g. Kokoro)
│
├── weights/                    # Actual downloaded/trained model files (never committed)
│   ├── tts/
│   │   ├── pocket_tts/         # Local/offline Pocket TTS checkpoints & voice embeddings
│   │   └── <other TTS weights>
│   ├── stt/
│   │   ├── faster-whisper-large-v3-turbo-ct2/  # Faster-Whisper CT2 model weights
│   │   └── <other STT weights>
│   └── other/                  # Other model checkpoints
│
├── scripts/                    # Executable scripts and command-line entry points
│   ├── tts/
│   │   ├── launch_gui.py       # Desktop Audiobook Generator GUI launcher
│   │   ├── launcher.pyw        # Windows GUI bootstrap launcher
│   │   ├── test.py             # Voice cloning & inference test script
│   │   └── convert.py          # 24 kHz mono WAV conversion utility
│   └── stt/
│       ├── transcribe.py       # Faster-Whisper transcription script (portable repo-relative)
│       └── launch_asr_gui.py   # Standalone ASR Validator GUI launcher
│
├── pocket_tts/                 # Core Pocket TTS Python package (intact)
│   ├── models/                 # Internal neural architecture (FlowLM, Mimi, CUDA graphs)
│   ├── audiobook/              # Audiobook chunking and generator engine
│   ├── gui/                    # PyQt5 desktop application tabs and windows
│   └── ...
│
├── ASR/                        # Complete ASR verification and alignment subsystem (intact)
│   ├── asr_gui.py              # Tkinter ASR Validator GUI
│   ├── asr_validator.py        # ASR scoring and pipeline validation
│   ├── streaming_stage_one.py  # Stage-one streaming ASR
│   └── ...
│
├── tools/                      # Developer utilities, benchmarks, profiling, and probes
│   ├── benchmark_tts.py
│   ├── benchmark_two_stage_asr.py
│   └── ...
│
├── tests/                      # Automated test suite
│
├── Testinput/                  # Test input audio files and reference voices
│   ├── Swetabh_Input.wav
│   ├── Swetabh Input.m4a
│   └── arctic_a0005.wav
│
├── Output/                     # Generated audio files (ignored in git)
│   └── cloned_output.wav
│
├── docs/                       # Documentation and assets
├── requirements.txt            # Runtime dependencies
├── requirements_windows.txt    # Windows launcher requirements
├── README.md
├── LICENSE
└── THIRD_PARTY_LICENSES.md
```

### Faster-Whisper Model Weights

Faster-Whisper model weights are expected at:
```text
weights/stt/faster-whisper-large-v3-turbo-ct2/
    ├── config.json
    ├── model.bin
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── vocabulary.json
```

To run transcription using the local model and portable repository-relative paths:
```bash
python scripts/stt/transcribe.py --audio Testinput/Swetabh_Input.wav --device cuda --compute-type float16
```

---

## Requirements

| Item             | Requirement                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| **OS**           | Linux (primary) or Windows                                                                 |
| **Python**       | 3.12 or later                                                                              |
| **GPU**          | NVIDIA GPU recommended, 8 GB VRAM minimum, 12 GB+ preferred                                |
| **CUDA**         | CUDA 12.8-compatible drivers for GPU PyTorch wheels                                        |
| **Disk**         | Several GB for the venv, Pocket TTS weights, and Whisper/Parakeet models                   |
| **FFmpeg**       | Required for M4B and MP3 export (Windows launcher can install a private copy)              |
| **Hugging Face** | Accept the Pocket TTS model terms; a token is required for the gated voice-cloning weights |

CPU generation works but is far slower. ASR Stage 2 on the New/Parakeet path is GPU-only and reports a load failure instead of silently falling back to CPU.

---

## Install

### Linux (recommended)

From the repository root:

```bash
chmod +x install.sh
./install.sh
```

`install.sh` will:

1. Require Python 3.12+.
2. Create `./venv` if it does not exist.
3. Install PyTorch / torchvision / torchaudio. **CUDA wheels** are used when `nvidia-smi` works; otherwise CPU wheels.
4. Install `requirements.txt`.
5. Prefetch Pocket TTS weights, tokenizer, and built-in voice embeddings.

Useful flags:

```bash
./install.sh --cuda                 # force CUDA torch wheels
./install.sh --cpu                  # force CPU torch wheels
./install.sh --torch-index-url URL  # custom wheel index
./install.sh --skip-models          # deps only; download models later
```

If Hugging Face access is gated, open [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), accept the terms, sign in (`huggingface-cli login` or `HF_TOKEN`), then rerun `./install.sh`.

### Windows

1. Extract the release or clone the repository.
2. Double-click `launcher.pyw`.
3. First run creates the environment, installs dependencies, can fetch a private FFmpeg build, and walks through Hugging Face access.
4. Later runs open the GUI immediately.

Manual Windows install (if you prefer not to use the launcher):

```bat
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Hugging Face access

Voice cloning uses a **gated** model. A public fallback without cloning exists, but custom WAV voices need the gated weights.

1. Visit [https://huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts).
2. Create a free account if needed and click **Agree and access repository**.
3. Create a read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Fine-grained tokens need permission for gated repos.
4. Store the token (`huggingface-cli login`, or the `HF_TOKEN` environment variable).

---

## Run

### Main GUI (Audiobook Generator)

```bash
# Linux, after install.sh
source venv/bin/activate
python scripts/tts/launch_gui.py
# (or python launch_gui.py via root compatibility wrapper)
```

Windows: double-click `launcher.pyw` (root or `scripts/tts/launcher.pyw`), or:

```bat
venv\Scripts\python scripts\tts\launch_gui.py
```

Equivalent entry points:

```bash
python scripts/tts/launch_gui.py
python launch_gui.py
python -m pocket_tts.gui.main_window
```

`python -m pocket_tts` starts the original Kyutai CLI / local HTTP server, not this desktop studio.

### Standalone ASR Validator & Transcription

Run ASR GUI:
```bash
python scripts/stt/launch_asr_gui.py
# or
./ASR/run.sh
# or
venv/bin/python ASR/asr_gui.py
```

Transcribe Audio (Faster-Whisper):
```bash
python scripts/stt/transcribe.py --audio Testinput/Swetabh_Input.wav
```

---

## First launch

On first start the app may:

- Open **Pocket TTS - Setup** if weights are missing, with **Download TTS Model** and **Download Tokenizer**.
- Offer to **build whisper.cpp CUDA** if you select the `whisper_cpp` backend and a CUDA-capable device. Prefer **faster_whisper** unless you specifically need ggml.
- Run a one-time **ASR worker calibration** when a sample book is present. The suggested GPU-worker count becomes the default; the spinner can still override it per run.

The window maximizes when possible. Browse dialogs open at about 75% of the screen, hide dotfiles by default, and start on **All Files (\*)**.

---

## User manual — Generate Audiobook

This is the Main tab. Read it top to bottom the first time you run a book.

### File Selection

#### Text File

Displays the selected book path, or `No file selected`.

**Browse...** opens a file dialog for `.txt` (or any file). After a valid file is chosen, **Generate Audiobook** and **Add to Batch** become enabled.

Use UTF-8 plain text. Chapter headings such as `Chapter One` or `Part II` are detected for export when **Chapterize** is on. Manual pauses use `[0.15s]` or `[4s]` in the text (see [Text conventions](#text-conventions)).

#### Voice

| Choice                                                                | Meaning                                               |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| `alba (default)`                                                      | Built-in female default.                              |
| `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma` | Other built-in embeddings.                            |
| `Custom WAV...`                                                       | Opens a picker for your own reference WAV.            |
| `Custom: filename.wav`                                                | Last custom voice, restored if the file still exists. |

**Browse...** also picks a custom WAV and adds it to the list. The WAV is a **voice prompt**, not the book audio. A clean, dry, single-speaker clip works best.

Output folders are named from the book title plus the voice, under `Output/`.

### Header controls

#### Show Parameters

Checked by default. Uncheck to hide the eight parameter columns and give the Results log more height. Values stay in effect; only the panel is hidden. Long blocking work (for example a whisper.cpp CUDA build) may hide the panel automatically so log lines stay visible.

#### Save Configuration

Writes the current controls to `pocket_tts/config/default_config.yaml`.

Important distinction:

- **Generate** and **Add to Batch** always use the **live** spinner and checkbox values, even if you have not saved.
- Saved YAML is what the next session loads.
- ASR GPU worker count is applied to the current run either way; it is written to disk only when you save.

#### Meta Data

Green button. Opens **Export Meta Data** (documented [below](#user-manual--export-meta-data)). Tags apply to the next Main or Batch export and persist if you then click **Save Configuration**.

### TTS Core Parameters

How the neural model samples each chunk.

| Control              | Range                | What it does                                                                                                                                     |
| -------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Temperature**      | 0.00–1.50, step 0.05 | Randomness. Lower (0.5–0.7) is more stable and consistent. Higher (0.9–1.2) is livelier but more likely to glitch. Start near 0.6–0.7.           |
| **EOS Threshold**    | −10.0–0.0            | How strongly the model must “see” end-of-sequence before stopping. More negative keeps talking longer. Closer to zero stops sooner and can clip. |
| **Frames After EOS** | 0–10                 | Extra frames after end-of-speech so the last syllable is not cut. 1–3 is usually enough.                                                         |

Click **?** in the column header for the in-app explanation.

### Chunking Settings

How the book is cut into TTS units. Smaller chunks are easier for the model and for ASR; too small sounds choppy and increases overhead.

| Control               | What it does                                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Mode: sentence**    | Pack full sentences until **Min Words** is met. A chunk may exceed the floor; it never splits mid-sentence. Usual choice for narration plus ASR.                                           |
| **Mode: paragraph**   | One chunk per paragraph (`\n\n`). **Min Words** is ignored and disabled. Long paragraphs can stress quality.                                                                               |
| **Mode: word_target** | **Min Words** is a soft floor; **Target Words** is a hard cap. Full sentences are packed up to the cap. A single over-long sentence stays whole. Blank lines do **not** force a new chunk. |
| **Min Words**         | Soft merge floor in sentence and word-target modes (1–100).                                                                                                                                |
| **Target Words**      | Hard cap for word-target only (1–999). Hidden/inactive semantics still store a value for other modes.                                                                                      |

Chapter and structure detection still run; they affect metadata and export, not only the split.

### Pause Durations (ms)

Silence **between** finished chunks when the book is assembled. These do not change how the model speaks inside a chunk.

| Control             | Range                | Typical use                                             |
| ------------------- | -------------------- | ------------------------------------------------------- |
| **Sentence End**    | 0–2000 ms, step 50   | Gap after a sentence-boundary chunk. Default often 400. |
| **Paragraph Break** | 0–5000 ms, step 100  | Longer gap between paragraphs. Default often 700.       |
| **Chapter Start**   | 0–10000 ms, step 500 | Gap at chapter-scale boundaries. Default often 1000.    |

If **Enable punctuation pauses** is on, these three values are forced to **zero for that run**. Intra-chunk rhythm then comes from punctuation injection and manual `[Xs]` markers, not from these join gaps.

### Quality Settings

| Control                      | What it does                                                                                                                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LSD Steps**                | Latent refinement steps, 1–30. Higher is usually cleaner and slower. Mid-range 5–10 is a common quality/speed balance. The panel notes this range.                                                                               |
| **Enable speed variation**   | When on, slight speaking-rate changes (emotion / style) reduce a metronomic pace. When off, pacing stays uniform.                                                                                                                |
| **Enable emotion detection** | When on, DistilRoBERTa scores each chunk (joy, anger, sadness, fear, surprise, disgust, neutral) and maps that to temperature and speed. When **off**, the emotion model is not loaded; every chunk uses neutral TTS parameters. |

Turn emotion off for a flatter, more consistent narrator, or when you want to save a little CPU.

### Pause Injection (s)

Converts punctuation into **pause events**. Each text fragment is generated as a complete waveform; exact digital silence is then placed between fragments. The spoken audio is never cut inside a finished waveform automatically by the program.

| Control                       | What it does                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Enable punctuation pauses** | Master switch for **automatic** punctuation pauses only. Manual `[Xs]` markers always work, even when this is off. |
| **Period (.)**                | Silence after a period, 0.00–3.00 s.                                                                               |
| **Exclamation (!)**           | Silence after `!`.                                                                                                 |
| **Question (?)**              | Silence after `?`.                                                                                                 |
| **Comma (,)**                 | Usually the shortest automatic pause.                                                                              |
| **Ellipsis (...)**            | Pause for ellipsis.                                                                                                |
| **Em Dash (--)**              | Pause for em dash / double hyphen.                                                                                 |
| **Semicolon (;)**             | Pause for semicolon.                                                                                               |
| **Colon (:)**                 | Pause for colon.                                                                                                   |

Set a mark to `0.00 s` to skip it. Unchecking the master switch disables the spinners.

Manual examples that always work: [xs] may be placed direct in input text to add additonal pauses.  For instance [1s] Chapter One. will add 1 sec delay before Chapter One is spoken.

```text
Chapter, [4s] One.
She waited. [0.15s] Then she spoke.
```

### Audiobook Export

Only **checked** formats are built. You can skip the full WAV if you only need M4B or MP3.

| Control                          | What it does                                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **M4B**                          | One chapter-friendly M4B from the chunk WAVs. Does not require writing a full WAV. Needs FFmpeg.                                      |
| **MP3**                          | With Chapterize: one MP3 per chapter under `chapters/`. Without: one full-book MP3. Needs FFmpeg.                                     |
| **WAV**                          | Full stitched WAV. Turn off for a faster finish when only M4B/MP3 are needed. WAV can carry text tags but **cannot embed cover art**. |
| **Chapterize**                   | Build a chapter table of contents. Off = one complete book.                                                                           |
| **Chapter strategy**             | How chapters are planned (only used when Chapterize is on). See below for details.                                                    |
| **Minutes / Fallback / Maximum** | Minute length used by the strategies below. `0` means “not used” or “no maximum”, depending on strategy.                              |
| **Normalization**                | `none`, `peak`, `loudness`, or `simple`.                                                                                              |

**Chapter strategies**

| Strategy                        | Behavior                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Headings only**               | Use detected Part/Chapter headings. If none exist, the whole book is one chapter. The minutes spinner is unused.          |
| **Headings, otherwise minutes** | Keep heading chapters intact. If **no** heading exists, split near paragraph or sentence ends using the fallback minutes. |
| **Headings + maximum duration** | Use headings, then split any chapter longer than the maximum minutes near paragraph or sentence ends.                     |

A note under the strategy explains the active mode. Minute-based splits prefer paragraph ends, then sentence ends.

### Performance Settings

| Control         | What it does                                                                                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Device**      | `auto` prefers CUDA when available. `cuda` forces GPU. `cpu` forces CPU. The tooltip names the detected GPU.                                                                                          |
| **Max Workers** | Parallel TTS workers. The upper limit is estimated from VRAM (~1.5 GB per worker after a 2 GB reserve) and CPU cores. More workers = faster book and more memory. Lower this on out-of-memory errors. |

On an 8 GB card, a few workers plus batching usually beats maxing the spinner. Leave VRAM headroom if post-generation ASR is enabled. For CPU experiment, after a certain point more CPU's adds minimal increase in speed.  On 8 physicaly cores 4 is usually best and leave the system responsive enough to use for other tasks while generation is happening. Using 8 CPU's  on the same system will likely cripple the OS and stall or crash.  RAM is also a consideration.  

### ASR Quality Control

After (and optionally during) TTS, the app can listen to each chunk and compare it with the source. Results are binary: **PASS** or **FAIL**. There is no review state.

ASR happens in 2 stages.  Stage 1 runs base model or parakeet by default. These are FAST but find lots of fails.  Stage 2 runs Faster-Whisper Medium model by default ONLY on the reported fails from stage 1 not all wav files. Then regeneration only happens on files that Stage 2 fails. This is faster than running medium on thousands of chunks and acts as a funnel 6k chunks -> Stage 1 189 fails -> Stage 2 35 fails -> only regenerates 35 wavs.  

| Control                             | What it does                                                                                                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Enable ASR Quality Control**      | Master switch. Off = no listening and no ASR-driven regeneration.                                                                                                                                            |
| **Stage 1 pipeline**                | Which engine listens first, and when.                                                                                                                                                                        |
| **Second-stage ASR model**          | Independent Faster-Whisper verifier (Parakeet / New path). Default `medium`. GPU only.                                                                                                                       |
| **Run forced-alignment diagnostic** | New/Parakeet path only. Extra evidence; **does not** change pass/fail or regeneration.                                                                                                                       |
| **ASR Backend**                     | Stage 1 / fallback engine when the pipeline allows it: `parakeet`, `faster_whisper`, `whisper_cpp`.                                                                                                          |
| **ASR Model**                       | Size for Faster-Whisper / whisper.cpp (`tiny` … `large-v3-turbo`, plus `distil-*`). Distil models are real on Faster-Whisper; whisper.cpp maps them to the nearest full size.                                |
| **GPU ASR workers (after gen)**     | Post-gen worker budget. `0` turns post-gen workers off. Enabled for **Post-gen Faster-Whisper**. Faster-Whisper: CPU budget around one GPU model (try ~8). whisper.cpp: concurrent GPU copies (1–3 on 8 GB). |
| **Language**                        | Recognition language. Must match the book. Default English.                                                                                                                                                  |
| **Threshold**                       | Minimum similarity to pass (0.00–1.00). Higher = stricter, more retries. Typical working range is about 0.60–0.85.                                                                                           |
| **Max Retries**                     | How many times a failed chunk may be regenerated. **`0` = report only** — ASR writes reports and never resynthesizes.                                                                                        |
| **Temp Decrement**                  | How much to lower TTS temperature on each retry (0.01–0.50). Larger steps move faster toward flatter, more stable speech.                                                                                    |

**Stage 1 pipeline choices**

| Label                         | Meaning                                                                                                                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Post-gen Faster-Whisper**   | After TTS finishes, Faster-Whisper is Stage 1 base model by default.  GPU-worker spinner applies.                                                                                                                                                                           |
| **Post-gen Parakeet**         | After TTS, NVIDIA Parakeet is Stage 1; selected Faster-Whisper model is Stage 2. Writes `asr_new_*` reports.                                                                                                                                                                |
| **During-TTS Faster-Whisper** | Collects Stage 1 evidence while chunks are generated (shadow collection). Post-gen ASR remains authoritative for reports and regeneration. NOTE: Does NOT finish when TTS generations does usually.  Only a 20-30% gain over post-gen method. May be faster on newer GPU's. |
| **During-TTS Parakeet**       | Same schedule using Parakeet for the shadow Stage 1 pass. NOTE: same as above.                                                                                                                                                                                              |

Selecting a Parakeet pipeline disables **ASR Backend** and **ASR Model** (Parakeet is fixed for Stage 1) and enables **Second-stage ASR model** plus the alignment checkbox.

Comparator notes (no per-book alias lists):

- Numbers, times, currency, and IDs become positional slots, so `0900` and “zero nine hundred” can match.
- Standard titles (`Dr.` → doctor), letter sequences (`S-O-B` / `S.O.B.`), and many spoken spelling variants are normalized. Use https://github.com/danneauxs/BookFix to preprocess your text files.
- Negations, missing content, extra phrases, and structural changes still fail.  Load fails in Regeneration Tab to examine all failed chunks and fix them then rebuild output files if needed.  Or accept a few imperfect audio chunks and enjoy your audio file. 

### Generation Progress

| Control                             | What it does                                                                                                                                                                                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chunks: n/n**                     | Completed vs total chunks.                                                                                                                                                                                                                      |
| **Elapsed**                         | Wall time for the current run.                                                                                                                                                                                                                  |
| **ETA**                             | Estimated remaining generation time.                                                                                                                                                                                                            |
| **Completion timings** (right side) | After a successful run: `1-HH:MM:SS` (ASR Stage 1), `2-HH:MM:SS` (Stage 2), then chunk-generation realtime factor and time, then end-to-end realtime factor and time. Cleared when the next run starts. Missing stage times show as `00:00:00`. |
| Progress bar                        | 0–100% of the current generation.                                                                                                                                                                                                               |
| **Generate Audiobook**              | Start the pipeline for the selected text and voice. Disabled until a file is chosen, and while a run is active.                                                                                                                                 |
| **Add to Batch**                    | Snapshot this file + voice + **all live settings** onto the Batch tab. Does not start generation.                                                                                                                                               |
| **Stop**                            | Cancel the current run. Space/Enter will not trigger it (avoids accidental cancel).                                                                                                                                                             |
| **Play Last Audio**                 | Play the last finished export after a successful run.                                                                                                                                                                                           |

A new Generate run **clears** `audio_chunks/` and `text_chunks/` for that book and removes stale top-level ASR JSON/log files in the TTS folder. Historical `run*.log` files are kept.

Status bar at the bottom of the window shows `Ready` when idle.

### Results

Read-only session log: selected paths, ASR summary, progress, errors, and save confirmations. Dark terminal styling. This is the first place to look when a run fails.

---

## User manual — Export Meta Data

Opened by **Meta Data** on the Generate tab.

| Field               | Purpose                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Author**          | Book author.                                                                                                                             |
| **Series**          | Series name.                                                                                                                             |
| **Series #**        | Volume number in the series.                                                                                                             |
| **Title**           | Audiobook title.                                                                                                                         |
| **Composer**        | Composer tag.                                                                                                                            |
| **Year**            | Publication year.                                                                                                                        |
| **Genre**           | Genre tag.                                                                                                                               |
| **Artist**          | Defaults to `Dnxs Spokenword PocketGPU` if left empty.                                                                                   |
| **Description**     | Longer blurb / comment.                                                                                                                  |
| **Cover image**     | Read-only path plus thumbnail.                                                                                                           |
| **Browse Image...** | Pick a JPEG or PNG. The dialog starts in the selected book’s folder.                                                                     |
| **Save / Cancel**   | Save applies to the **next** export (Main and Batch snapshots). Click **Save Configuration** on the Main tab to persist across sessions. |

Cover art embeds in **M4B and MP3**. WAV receives text metadata only.

---

## User manual — Regenerate Chunks

Use this tab after a book already exists under `Output/`.

### TTS Folder Selection

**TTS Folder** shows `No folder selected` until you browse.

**Browse...** should target the **TTS** directory, for example:

```text
Output/My Book Title/TTS
```

Required contents:

- `audio_chunks/chunk_00000.wav` …
- `text_chunks/audiobook.chunks.json`

If either is missing, a warning explains what to select.

### Search Chunks

| Control              | What it does                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Keyword**          | Case-insensitive search of chunk text. Enter or **Search**.                                                                                                                          |
| **Chunk ID**         | Jump to a numeric index (`203` → `chunk_00203`). Enter or **Go**.                                                                                                                    |
| **Load Fail Report** | Load `asr_new_medium_failures.json` from the selected TTS folder (New-pipeline Medium confirmed failures). Legacy list-shaped reports in that same filename shape are also accepted. |

Search controls stay disabled until a valid TTS folder is loaded.

### Search Results

Header shows `Results: None` or a title such as `ASR Failed Chunks (12)`. Click a row (`chunk_00042: preview…`) to load it.

### Chunk Editor

| Control                | What it does                                                                     |
| ---------------------- | -------------------------------------------------------------------------------- |
| **Chunk: chunk_NNNNN** | Selected index.                                                                  |
| **Text**               | Editable source for the next regeneration. Fix typos or add `[Xs]` here.         |
| **Emotion**            | Neutral, Joy, Anger, Sadness, Fear, Surprise, Disgust. Used when you regenerate. |
| **Confidence**         | Stored emotion confidence (read-only).                                           |
| **View Details**       | Emotion scores, optional override, and the TTS parameters that will be used.     |
| **Voice**              | Voice for this regeneration (built-in or from the book folder).                  |
| **Play Original**      | Play `audio_chunks/chunk_NNNNN.wav`.                                             |
| **Regenerate**         | Synthesize a new take into a temporary file. Does not overwrite until you Save.  |
| **Play New**           | Play the temporary regeneration.                                                 |
| **Save**               | Replace the original chunk WAV (original is backed up) and update metadata.      |
| Status line            | `Ready`, progress, or success/failure.                                           |

If a fail report is loaded, failure details (original vs ASR transcript, score, explanation) are available for that chunk.

### Audiobook Rebuild

**Rebuild Audiobook** restitches every current chunk WAV using the **live Main-tab** export settings:

- M4B / MP3 / WAV checkboxes
- Chapterize, chapter strategy, minutes
- Normalization and metadata/cover

It does **not** use a separate checkbox on this tab. If no export format is checked on Generate, rebuild warns you to select one first.

The output name appears as `Output: filename`. A dialog lists created files and chapter count.

---

## User manual — Batch Processing

### Files to Process

List of queued jobs. Each line is one book; the log shows the reserved output folder (`_batch2`, `_batch3`, … if the default folder already exists).

| Button                  | What it does                                                                                                                                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Add Folder**          | Pick a directory. Every `*.txt` is queued **sorted by name**, each with the **current Main-tab** voice and settings. Needs the Generate tab. Disabled in effect while a batch is running (you must stop first). |
| **Remove**              | Delete the selected job.                                                                                                                                                                                        |
| **Clear All**           | Empty the queue.                                                                                                                                                                                                |
| **Move Up / Move Down** | Change run order.                                                                                                                                                                                               |

You can also press **Add to Batch** on Generate to queue the currently selected file without leaving that tab.

### Overall Progress

| Display         | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| **File: n / n** | Books finished vs queued.                                        |
| Overall bar     | Progress across the whole queue.                                 |
| **Current**     | Active book, or `ASR check…` / `Regeneration…` / `n / m chunks`. |
| Current bar     | Progress inside the active book.                                 |
| **Elapsed**     | Queue wall time (`MM:SS`).                                       |
| **ETA**         | Estimated remaining queue time.                                  |

### Run controls

| Button          | What it does                                                  |
| --------------- | ------------------------------------------------------------- |
| **Start Batch** | Run the queue in order. Green.                                |
| **Pause**       | Pause after the current safe point. Label becomes **Resume**. |
| **Stop**        | Cancel the batch. Red.                                        |

### Log

Read-only, dark, green monospace. Records queue, start, ASR phases, and errors.

Settings rule: **the queue is frozen**. Changing Generate after **Start Batch** does not rewrite jobs already listed.

---

## User manual — PocketGPU ASR Validator

Companion window: **PocketGPU ASR Validator**. It does not generate speech. It re-listens to existing chunks.

Launch with `./ASR/run.sh` or `venv/bin/python ASR/asr_gui.py`.

### Book / TTS folder

| Control      | What it does                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------ |
| **Selected** | Read-only path.                                                                                  |
| **Browse…**  | Choose a book folder or its `TTS` subfolder. The tool resolves `audio_chunks` and text sidecars. |
| Path hint    | Shows how the folder was interpreted.                                                            |

### ASR settings

| Control               | What it does                                                                                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**           | `faster_whisper` (recommended) or `whisper_cpp`. whisper.cpp + GPU may offer to compile CUDA support (several minutes).                                                                                         |
| **Model**             | Whisper size / distil variant.                                                                                                                                                                                  |
| **Device**            | `cpu`, `cuda`, or auto-style device choice stored from last session.                                                                                                                                            |
| **Workers**           | 1–16 parallel validation workers.                                                                                                                                                                               |
| **Language**          | Recognition language label.                                                                                                                                                                                     |
| **Threshold**         | Pass cutoff, 0.00–1.00.                                                                                                                                                                                         |
| **Process ASR Fails** | When checked, validate only rows from a failure JSON instead of every chunk.                                                                                                                                    |
| **Failure report**    | Path to that JSON.                                                                                                                                                                                              |
| **Browse…** (report)  | Defaults to `*.json`; you can switch to `*.*`. Accepts a JSON **list** or `{ "records": [ ... ] }`. Folder scan prefers `asr_new_medium_failures.json`, then `asr_new_failures.json`, then `asr_failures.json`. |
| **Log Filter**        | `Both`, `Pass Only`, or `Fail Only`.                                                                                                                                                                            |

### Actions and status

| Control             | What it does                                               |
| ------------------- | ---------------------------------------------------------- |
| **Run Validation**  | Start. Logs are written into the selected book/TTS folder. |
| **Stop**            | Request cancel.                                            |
| Status line         | `Ready` or current phase.                                  |
| Progress bar        | Chunk progress.                                            |
| Download line / bar | Model download progress on first use of a size.            |
| Stats line          | Pass / fail counts.                                        |
| **Log**             | Scrollable run log.                                        |

Use this tool to re-score a book, or to drive a focused pass over a previous fail list, without starting TTS again.

---

## Typical workflows

### One book, listen as you go

1. Generate tab → **Browse...** the `.txt`.
2. Pick a voice.
3. Leave **Show Parameters** on and review chunking (sentence + a small Min Words is a solid start).
4. Check **WAV** (and **M4B** if you use an audiobook player).
5. Optional: **Enable ASR Quality Control**, threshold around 0.60–0.75, **Max Retries** 2–3.
6. **Meta Data** if you want tags/cover.
7. **Generate Audiobook**.
8. When finished, **Play Last Audio**, or open `Output/<Book>/`.

### Overnight shelf of books

1. Set voice, export, and ASR on Generate (the settings you want for **all** of them).
2. Either **Add to Batch** per file, or on Batch use **Add Folder**.
3. Confirm the queue and output folder names in the Batch log.
4. **Start Batch**.
5. Use **Pause** if you need the GPU; **Stop** to abort.

### Repair a few bad sentences

1. Finish (or stop) the original run.
2. Regenerate tab → browse `Output/<Book>/TTS`.
3. **Load Fail Report** or search by keyword / ID.
4. Edit text if needed, **Regenerate**, **Play New**, **Save**.
5. Set export formats on Generate, then **Rebuild Audiobook**.

### Report-only quality audit

1. Enable ASR.
2. Set **Max Retries** to **0**.
3. Generate (or run the ASR Validator on an existing TTS folder).
4. Inspect `asr_new_medium_failures.json` / `asr_failures.json` and the investigation logs. No audio is rewritten.

---

## Output layout

```text
Output/
└── <BookTitle>/
    ├── <filename> [<voice>].wav          # if WAV export is on
    ├── <filename> [<voice>].m4b          # if M4B export is on
    ├── chapters/                         # if MP3 + Chapterize
    └── TTS/
        ├── audio_chunks/
        │   ├── chunk_00000.wav
        │   └── Failed/                   # rebuilt empty at each new Generate
        ├── text_chunks/
        │   ├── chunk_00000.txt
        │   └── audiobook.chunks.json
        ├── run_YYYYMMDD_HHMMSS.log       # kept across runs
        ├── asr_new_medium_failures.json  # New pipeline confirmed fails
        ├── asr_new_pipeline_report.json
        └── asr_failures.json             # Legacy Stage 1 list (when that path runs)
```

A new Generate on the same book wipes `audio_chunks` and `text_chunks` and removes leftover top-level ASR JSON/logs **except** `run*.log`.

Batch collisions get `..._batch2`, `..._batch3`, and so on.

---

## Text conventions

| In the `.txt` file                       | Result                                                                                                                            |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Normal punctuation                       | Spoken as written. If **Enable punctuation pauses** is on, configured silence is inserted at those marks.                         |
| `[4s]`, `[0.15s]`                        | Exact digital silence, always, even when punctuation pauses are off. Prefer `Chapter, [4s] One.` over gluing the marker to words. |
| `Chapter One` / `Part II` / `Book Three` | Detected as headings for chapterized export. Markers inside a heading are stripped from the displayed title.                      |
| Blank line (`\n\n`)                      | Paragraph break. In **paragraph** mode this is also a chunk boundary.                                                             |
| Very short lines                         | In **sentence** mode they merge with neighbors until **Min Words**.                                                               |

Do not put the pause marker in the middle of a word.

---

## Configuration

Authoritative GUI defaults: `pocket_tts/config/default_config.yaml`.

**Save Configuration** updates that file. A run does **not** require a save; live controls win.

Groups you will recognize from the GUI:

| YAML section                              | GUI column                   |
| ----------------------------------------- | ---------------------------- |
| `tts_core`                                | TTS Core Parameters          |
| `chunking`                                | Chunking Settings            |
| `pauses.base_durations`                   | Pause Durations              |
| `quality` / `speed_variation` / `emotion` | Quality Settings             |
| `pause_injection`                         | Pause Injection              |
| `m4b` (including `metadata`)              | Audiobook Export + Meta Data |
| `device` / `parallel`                     | Performance Settings         |
| `asr_quality_control`                     | ASR Quality Control          |

---

## Troubleshooting

### Out of memory / CUDA OOM

- Lower **Max Workers**.
- Prefer sentence chunking with a moderate **Min Words**.
- Use Faster-Whisper `base` or `small` for Stage 1; keep Stage 2 at `medium` unless you have VRAM to spare.
- Lower **GPU ASR workers (after gen)** (try 2–4 on 8 GB).
- Close other GPU applications.
- If punctuation pauses are on, every mark becomes its own TTS fragment — that increases work. Turn the switch off or zero unused marks.

### Generation is slow

- Confirm **Device** is `cuda` or `auto` and the tooltip shows your GPU.
- Raise **Max Workers** only until VRAM is comfortable.
- Lower **LSD Steps** before cutting workers if quality is already acceptable.
- ASR after a long book can take minutes; that time is not TTS slowness.

### ASR reports many failures

- Confirm **Language** matches the book.
- Lower **Threshold** slightly (for example 0.60–0.68) before assuming the voice is wrong.
- Read the failure JSON: many “fails” are spelling/normalization, not missing sentences.
- Set **Max Retries** to 0, inspect reports, then regenerate only real errors on the Regenerate tab.

### Custom voice ignored / falls back

- Accept the gated model terms and provide a Hugging Face token with gated-repo access.
- Confirm the WAV path still exists (Custom entries store a full path).

### M4B / MP3 missing

- Check the corresponding export checkbox.
- Install FFmpeg (Linux package, or let the Windows launcher fetch its private copy).
- Cover art is JPEG/PNG only and will not appear inside WAV.

### Regenerate tab will not load a folder

- Select the **TTS** folder, not the parent book folder.
- Both `audio_chunks/` and `text_chunks/audiobook.chunks.json` must exist.

### Load Fail Report says the file is missing

- The tab looks for `asr_new_medium_failures.json` in that TTS folder.
- Run a New-pipeline (Parakeet) ASR pass first, or use **PocketGPU ASR Validator**, which can open other JSON names.

### During-TTS ASR looks incomplete

- During-TTS Stage 1 is **shadow collection**. Final reports and regeneration still follow the post-gen path. Read the Results line that states the selected Stage 1 pipeline.

### Browse dialogs look empty or tiny

- Dotfiles are hidden on purpose. The sidebar is resized after the dialog opens so the folder tree stays readable.

## Responsible Use Notice

This software is intended for lawful and responsible text-to-speech applications, including accessibility, personal use, research, education, and the creation of audio from content that you own or are authorized to use.

Users are responsible for ensuring that all text, books, documents, voices, recordings, and other materials processed with this software are used in accordance with applicable laws and the rights of their respective owners.

This software is **not intended to facilitate copyright infringement, piracy, unauthorized distribution of copyrighted works, fraud, impersonation, deception, harassment, or other unlawful or abusive activity**.

Do not use this software to reproduce or distribute copyrighted material unless you own the necessary rights or have permission or another lawful basis to do so. Likewise, do not use synthesized or replicated voices in a manner that falsely represents another person, misleads others about the origin of audio, or violates applicable rights or laws.

The developers do not endorse or encourage misuse of this software. **You are solely responsible for the content you process, the audio you generate, and how that audio is used or distributed.**

**Use this software responsibly and respect copyright, licensing terms, privacy, and the rights of others.**---

## License and credits

DNXS Spoken Word · Pocket-TTS GPU Edition is licensed under the
GNU General Public License v3.0. See LICENSE.

Copyright © 2026 danneauxs.

This project is built upon and uses third-party open-source software
and machine-learning models. Those components remain subject to their
respective licenses. See THIRD_PARTY_LICENSES.md.

Pocket TTS model and core synthesis:
Kyutai Pocket-TTS

Pocket TTS model and core synthesis: [Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)  
Authors: Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez (* equal contribution).

This repository adds the GPU desktop studio: parallel generation,
emotion-aware narration, punctuation and manual pauses, two-stage ASR
quality control, batch queue, chunk repair, and audiobook export.

**DNXS Spoken Word · Pocket-TTS GPU Edition**
