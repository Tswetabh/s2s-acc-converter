#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
TORCH_MODE="auto"
TORCH_INDEX_URL=""
SKIP_MODELS=0

usage() {
    cat <<'EOF'
Usage: ./install.sh [--auto|--cuda|--cpu] [--torch-index-url URL] [--skip-models]

--auto              Pick CUDA if nvidia-smi is available, else CPU.
--cuda              Force CUDA torch wheels.
--cpu               Force CPU torch wheels.
--torch-index-url   Override torch wheel index URL completely.
--skip-models       Install deps only; skip Pocket model prefetch.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto) TORCH_MODE="auto" ;;
        --cuda) TORCH_MODE="cuda" ;;
        --cpu) TORCH_MODE="cpu" ;;
        --torch-index-url)
            [[ $# -ge 2 ]] || { echo "Missing value for --torch-index-url" >&2; exit 1; }
            TORCH_INDEX_URL="$2"
            shift
            ;;
        --skip-models) SKIP_MODELS=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

check_python() {
    "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python {sys.version.split()[0]} too old; need 3.12+")
PY
}

resolve_venv_python() {
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        printf '%s\n' "$VENV_DIR/bin/python"
        return 0
    fi
    if [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
        printf '%s\n' "$VENV_DIR/Scripts/python.exe"
        return 0
    fi
    if [[ -x "$VENV_DIR/Scripts/python" ]]; then
        printf '%s\n' "$VENV_DIR/Scripts/python"
        return 0
    fi
    return 1
}

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

choose_torch_index_url() {
    if [[ -n "$TORCH_INDEX_URL" ]]; then
        printf '%s\n' "$TORCH_INDEX_URL"
        return 0
    fi

    case "$TORCH_MODE" in
        cpu)
            printf '%s\n' "https://download.pytorch.org/whl/cpu"
            ;;
        cuda)
            printf '%s\n' "https://download.pytorch.org/whl/cu128"
            ;;
        auto)
            if has_nvidia_gpu; then
                printf '%s\n' "https://download.pytorch.org/whl/cu128"
            else
                printf '%s\n' "https://download.pytorch.org/whl/cpu"
            fi
            ;;
        *)
            echo "Unknown torch mode: $TORCH_MODE" >&2
            exit 1
            ;;
    esac
}

install_torch() {
    local index_url="$1"

    echo "Installing torch trio from: $index_url"
    if ! "$VENV_PY" -m pip install torch torchvision torchaudio --index-url "$index_url"; then
        if [[ "$TORCH_MODE" == "auto" && "$index_url" != "https://download.pytorch.org/whl/cpu" ]]; then
            echo "CUDA torch install failed. Falling back to CPU wheels."
            "$VENV_PY" -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
        else
            return 1
        fi
    fi
}

prefetch_pocket_models() {
    "$VENV_PY" - <<'PY'
from pathlib import Path

import yaml

from pocket_tts.utils.utils import PREDEFINED_VOICES, download_if_necessary

try:
    root = Path.cwd()
    config_path = root / "pocket_tts" / "config" / "b6369a24.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    targets = [
        config["weights_path"],
        config["weights_path_without_voice_cloning"],
        config["flow_lm"]["lookup_table"]["tokenizer_path"],
    ]
    targets.extend(PREDEFINED_VOICES.values())

    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        local_path = download_if_necessary(target)
        print(f"{target} -> {local_path}")
except Exception as exc:
    raise SystemExit(
        "Pocket model prefetch failed. If Hugging Face access is gated, "
        "accept Pocket TTS terms on Hugging Face and rerun install.sh."
    ) from exc
PY
}

echo "Pocket TTS install"
echo "Root: $ROOT_DIR"
echo

check_python

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if ! VENV_PY="$(resolve_venv_python)"; then
    echo "Could not find venv Python under $VENV_DIR" >&2
    exit 1
fi

"$VENV_PY" -m pip install --upgrade pip setuptools wheel

TORCH_INDEX_URL_TO_USE="$(choose_torch_index_url)"
install_torch "$TORCH_INDEX_URL_TO_USE"

echo "Installing project requirements"
"$VENV_PY" -m pip install -r "$REQ_FILE"

if [[ "$SKIP_MODELS" -eq 0 ]]; then
    echo "Prefetching Pocket models and checkpoints"
    prefetch_pocket_models
else
    echo "Skipping model prefetch by request"
fi

echo
echo "Done."
echo "Activate with:"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "  source venv/bin/activate"
else
    echo "  venv\\Scripts\\activate"
fi
