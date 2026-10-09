# Other Model Integrations

This directory is designated for future model implementations, adapters, and wrappers
(e.g., Kokoro TTS, ChatterNT, NeMo Canary/Parakeet extensions, etc.).

Each model integration should reside in its own subdirectory with an adapter module:
```
models/other/<model_name>/
    ├── __init__.py
    └── adapter.py
```
Model weights must be stored under `weights/other/<model_name>/` rather than in this source directory.
