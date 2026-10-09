#!/bin/bash
# ASR Validator Launcher Script with Auto Virtual Environment Management

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           ASR Validation Tool - Launcher                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running from correct directory
if [ ! -f "asr_gui.py" ]; then
    echo "❌ Error: asr_gui.py not found in current directory"
    echo "   Script directory: $SCRIPT_DIR"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "✅ Running from: $SCRIPT_DIR"
echo ""

# Check Python availability
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found"
    echo "   Please install Python 3.8 or higher"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "✅ Found: $PYTHON_VERSION"
echo ""

# Prefer MAIN project venv (single env); fall back to legacy ASR/venv
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    VENV_DIR="$ROOT_DIR/venv"
    echo "✅ Using main project venv: $VENV_DIR"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    VENV_DIR="$SCRIPT_DIR/venv"
    echo "⚠️  Using legacy ASR/venv (prefer main project venv)"
else
    echo "❌ No venv found. Create main venv from repo root:"
    echo "   cd $ROOT_DIR && python3 -m venv venv && source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
else
    echo "❌ Error: Cannot find activation script in $VENV_DIR"
    read -r
    exit 1
fi
echo "✅ Virtual environment activated"
echo ""

echo "🔍 Checking ASR dependencies..."
python -c "from faster_whisper import WhisperModel; import rapidfuzz; import librosa" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing from repo root requirements.txt..."
    python -m pip install --upgrade pip -q
    python -m pip install -r "$ROOT_DIR/requirements.txt"
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        read -r
        exit 1
    fi
fi
echo "✅ Dependencies OK"
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              🚀 Launching PocketGPU ASR GUI                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Launch the batch GUI (model/backend/device/workers → logs in book folder)
python asr_gui.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Application closed successfully"
else
    echo "⚠️  Application exited with code: $EXIT_CODE"
fi

echo ""
echo "Press Enter to exit..."
read

exit $EXIT_CODE
