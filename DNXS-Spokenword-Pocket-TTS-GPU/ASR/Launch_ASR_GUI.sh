#!/usr/bin/env bash
# Double-click launcher: use MAIN project venv and start the ASR GUI.
#
# Usage:
#   Double-click this file (or PocketGPU_ASR_Validator.desktop)
#   Or from a terminal: ./Launch_ASR_GUI.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

GUI_APP="$SCRIPT_DIR/asr_gui.py"
LOG_FILE="$SCRIPT_DIR/launcher.log"

# Prefer main project venv; fall back to legacy ASR/venv
if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  VENV_PY="$ROOT_DIR/venv/bin/python"
  VENV_BIN="$ROOT_DIR/venv/bin"
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  VENV_PY="$SCRIPT_DIR/venv/bin/python"
  VENV_BIN="$SCRIPT_DIR/venv/bin"
  echo "WARNING: using legacy ASR/venv" >>"$LOG_FILE"
else
  VENV_PY=""
  VENV_BIN=""
fi

fail() {
  local msg="$1"
  echo "$(date -Iseconds) ERROR: $msg" | tee -a "$LOG_FILE" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="PocketGPU ASR" --width=420 --text="$msg" 2>/dev/null || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical "PocketGPU ASR" "$msg" 2>/dev/null || true
  fi
  if [[ -t 0 ]]; then
    read -r -p "Press Enter to close..."
  fi
  exit 1
}

echo "$(date -Iseconds) Launch request" >>"$LOG_FILE"

[[ -f "$GUI_APP" ]] || fail "asr_gui.py not found in:\n$SCRIPT_DIR"
[[ -n "$VENV_PY" && -x "$VENV_PY" ]] || fail "Main venv python not found.\nExpected:\n$ROOT_DIR/venv/bin/python\n\nCreate with: python3 -m venv venv && pip install -r requirements.txt"

export VIRTUAL_ENV="$(dirname "$(dirname "$VENV_PY")")"
export PATH="$VENV_BIN:${PATH:-}"
export PYTHONUNBUFFERED=1

if ! "$VENV_PY" -c "import faster_whisper, tkinter" >>"$LOG_FILE" 2>&1; then
  fail "Main venv missing ASR packages (faster_whisper / tkinter).\nSee:\n$LOG_FILE\n\nFix: pip install -r requirements.txt"
fi

echo "$(date -Iseconds) Starting: $VENV_PY $GUI_APP" >>"$LOG_FILE"
exec "$VENV_PY" "$GUI_APP" >>"$LOG_FILE" 2>&1
