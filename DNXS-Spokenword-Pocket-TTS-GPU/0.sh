#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_COMMAND='source venv/bin/activate || { echo "Could not activate venv."; read -r -p "Press Enter to close..."; exit 1; }; python launch_gui.py; status=$?; echo; echo "Pocket GPU exited with status $status."; read -r -p "Press Enter to close..."'

if command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal \
        --title="Pocket GPU" \
        --working-directory="$SCRIPT_DIR" \
        -- bash -ic "$RUN_COMMAND"
fi

if command -v xterm >/dev/null 2>&1; then
    exec xterm \
        -title "Pocket GPU" \
        -e bash -ic "cd \"$SCRIPT_DIR\" && $RUN_COMMAND"
fi

printf 'No supported terminal emulator found. Install gnome-terminal or xterm.\n' >&2
exit 1
