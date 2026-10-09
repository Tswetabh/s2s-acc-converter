# Licensing

This repository contains **original work by danneauxs** plus **third-party
software and models that keep their own licenses**. This file explains the
split. It does not relicense anyone else's work.

## Project's original code — GNU GPL v3

Copyright (C) 2026 danneauxs

The original code written for **DNXS Spoken Word · Pocket-TTS GPU Edition**
is licensed under the **GNU General Public License version 3**.

That includes, without claiming this list is exhaustive:

- The desktop Audiobook Generator GUI (`pocket_tts/gui/`, `launch_gui.py`,
  `launcher.pyw`, `0.sh`)
- Audiobook generation, export, metadata, and chapterization
  (`pocket_tts/audiobook/`, `pocket_tts/audio/`, related preprocessing)
- ASR quality-control pipeline and standalone ASR Validator (`ASR/`,
  `tools/run_asr_pipeline.py`, related tests)
- Installers and project documentation written for this edition
  (`install.sh`, `README.md`, this file)

The full GPL v3 text is in [LICENSE](LICENSE).

You can redistribute and/or modify that original code under GPL v3 (or,
at your option, any later version published by the Free Software
Foundation). It is distributed **without any warranty**.

GPL v3 applies **only** to that original code. It does **not** replace
the licenses of Pocket TTS, NeMo, Whisper, FFmpeg, PyTorch, or any other
upstream project or model.

## Third-party software and models — their licenses still apply

This application **uses** the following (and other) third-party projects.
They are **not** relicensed as GPL v3 by this repository. Their copyright
holders keep their rights. Use, redistribution, and model weights follow
**their** terms.

Typical components you will install or download when running this app:

| Component                                                                      | Typical license (confirm upstream) | Notes                                                                                                                                        |
| ------------------------------------------------------------------------------ | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)                 | Apache License 2.0                 | Core TTS model code and architecture this studio is built on. Apache 2.0 text is kept in [licenses/APACHE-2.0.txt](licenses/APACHE-2.0.txt). |
| [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) model weights    | Hugging Face model terms (gated)   | Accept the model terms on Hugging Face before download.                                                                                      |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Parakeet ASR)                   | Apache License 2.0                 | Used when the Parakeet Stage 1 path is selected.                                                                                             |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper)                    | MIT                                | Default Faster-Whisper ASR backend.                                                                                                          |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2)                          | MIT                                | Runtime used by faster-whisper.                                                                                                              |
| [openai/whisper](https://github.com/openai/whisper) and related model weights  | MIT (code); model cards apply      | Whisper-family checkpoints downloaded at runtime.                                                                                            |
| [pywhispercpp](https://github.com/absadiki/pywhispercpp) / whisper.cpp         | MIT                                | Optional `whisper_cpp` backend.                                                                                                              |
| [FFmpeg](https://ffmpeg.org/)                                                  | LGPL / GPL depending on the build  | Required for M4B and MP3 export.                                                                                                             |
| [PyTorch](https://github.com/pytorch/pytorch)                                  | BSD-style                          | GPU / CPU tensor runtime.                                                                                                                    |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) / Hub | Apache License 2.0                 | Emotion model and weight downloads.                                                                                                          |
| j-hartmann DistilRoBERTa emotion model                                         | See model card                     | Used only when emotion detection is enabled.                                                                                                 |
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)                     | GPL                                | Desktop GUI toolkit.                                                                                                                         |

Python packages listed in `requirements.txt` and `requirements_windows.txt`
each have their own license. Installed wheels and downloaded checkpoints
are **not** part of the GPL grant for this project's original source.

If a third-party file is vendored in this tree (for example Kyutai Pocket
TTS model/runtime modules under `pocket_tts/models/`, `pocket_tts/modules/`,
`pocket_tts/conditioners/`, and related Kyutai files), that file remains
under **its original license** unless the original authors say otherwise.

## Credits

Pocket TTS model and core synthesis:
[Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)

Authors: Manu Orsini\*, Simon Rouard\*, Gabriel De Marmiesse\*,
Václav Volhejn, Neil Zeghidour, Alexandre Défossez
(\* equal contribution).

This edition adds the GPU desktop studio around that engine.
