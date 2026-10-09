#!/usr/bin/env python3
"""
Pocket TTS Windows Launcher
First-run: installs PyTorch, transformers, and other dependencies (~2-3 GB)
Subsequent runs: launches Pocket TTS GUI immediately
Uses console output for progress feedback.
"""
import sys
import os
import subprocess
import shutil
import zipfile
import hashlib
import urllib.request
from pathlib import Path


INSTALL_DIR = Path(__file__).resolve().parent
BUNDLED_PYTHON = INSTALL_DIR / "python" / "python.exe"
BUNDLED_PYTHONW = INSTALL_DIR / "python" / "pythonw.exe"
LAUNCH_GUI = INSTALL_DIR / "launch_gui.py"
REQUIREMENTS_TXT = INSTALL_DIR / "requirements.txt"
SETUP_MARKER = INSTALL_DIR / ".setup_complete"
NOCONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Voice cloning uses a GATED HuggingFace model: the user must accept its
# terms on the website and supply a read token, otherwise the app silently
# falls back to the no-cloning public model. huggingface_hub finds the token
# via the HF_TOKEN env var or the token file below.
GATED_MODEL_URL = "https://huggingface.co/kyutai/pocket-tts"
TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"
# Fine-grained HF tokens have a permission checkbox for gated repos that is
# SEPARATE from the personal-namespace checkbox. A token can pass whoami
# and even read this repo's metadata while still 403ing on the actual file
# download if that box isn't checked - this exact URL (repo + pinned
# revision the app downloads) is what we probe to catch that case.
GATED_FILE_CHECK_URL = (
    "https://huggingface.co/kyutai/pocket-tts/resolve/"
    "427e3d61b276ed69fdd03de0d185fa8a8d97fc5b/tts_b6369a24.safetensors"
)
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
TOKEN_SKIP_MARKER = INSTALL_DIR / ".hf_token_skipped"
INSTRUCTIONS_FILE = INSTALL_DIR / "HF_SETUP_INSTRUCTIONS.txt"

# FFmpeg is required for M4B export and emotion-driven speed adjustment, but
# is never present on a clean Windows machine. Downloaded privately into the
# install folder (not system-wide) so no PATH edit / admin rights / reboot
# is ever needed - see plan notes on the HF_TOKEN PATH-caching bug this
# avoids repeating. Pinned to a specific version, not a "latest" pointer.
FFMPEG_DIR = INSTALL_DIR / "ffmpeg"
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/"
    "ffmpeg-8.1.2-essentials_build.zip"
)
FFMPEG_ZIP_INNER_PATH = "ffmpeg-8.1.2-essentials_build/bin/ffmpeg.exe"
# SHA256 of the extracted ffmpeg.exe itself (not the zip), checked after
# extraction so a corrupted/tampered download is caught before use.
FFMPEG_EXE_SHA256 = (
    "1326dde4c84ff1f96fe6b8916c5bed29e163e9b5dccf995f6f3db069d143ec5e"
)

INSTRUCTIONS_TEXT = """\
POCKET TTS - HUGGINGFACE SETUP INSTRUCTIONS
============================================

Voice cloning requires access to a GATED (but free) model on
HuggingFace. Keep this file open for reference - the setup window
will walk you through the same steps.

STEP A - ACCEPT THE MODEL TERMS
--------------------------------
1. Go to:  https://huggingface.co/kyutai/pocket-tts
2. If you do not have a HuggingFace account, you will be asked to
   sign up. It is free but required to use the model. Verify your
   email, then return to the model page.
3. Click the button: "Agree and access repository".

STEP B - CREATE AN ACCESS TOKEN
--------------------------------
1. Access Token Settings
   - Log in to your HuggingFace account.
   - Click on your profile picture (top-right corner).
   - Select Settings from the dropdown menu.
   - In the left-hand sidebar, click on Access Tokens.
   (Direct link:  https://huggingface.co/settings/tokens )

2. Create the Token
   - Click the "New token" button.
   - Token name: enter a descriptive name
     (e.g., "Personal Read Token").
   - Token type: select "Fine-grained".

3. Configure Permissions
   - After selecting "Fine-grained", you will see a list of
     permissions.
   - Locate the Repositories section.
   - Check the box labeled (REQUIRED - this is what lets Pocket TTS
     download the gated model; it is a DIFFERENT checkbox from the
     one below):
     "Read access to contents of all public gated repos you can
      access".
   - Also check the box labeled (optional, does not hurt to leave on):
     "Read access to contents of all repos under your personal
      namespace".
   - Click the "Create token" button at the bottom of the page.

4. Save Your Token
   - The token will appear on the screen. COPY IT IMMEDIATELY -
     HuggingFace will not show it to you again once you leave
     the page.
   - Store it in a secure location, such as a password manager.

STEP C - GIVE THE TOKEN TO POCKET TTS
--------------------------------------
Paste the token into the Pocket TTS setup window when asked.
It will be saved to your Windows environment automatically.

If you skipped this setup and want to enable voice cloning later:
delete the file ".hf_token_skipped" in the Pocket TTS install
folder and launch the app again.
"""


def _sha256_of(path):
    """Return the hex SHA256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_ffmpeg():
    """Download the pinned FFmpeg build and extract ffmpeg.exe only.

    Fetches the essentials zip to a temp file, verifies the extracted exe's
    SHA256 against the pinned value, and installs it to FFMPEG_EXE. Returns
    True on success, False on any failure (network, bad hash, etc.) so the
    caller can warn instead of silently continuing with no ffmpeg.
    """
    print("  Downloading FFmpeg (~110 MB, one-time)...")
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = FFMPEG_DIR / "_ffmpeg_download.zip.tmp"
    try:
        urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, tmp_zip)

        with zipfile.ZipFile(tmp_zip) as zf:
            with zf.open(FFMPEG_ZIP_INNER_PATH) as src, \
                 open(FFMPEG_EXE, "wb") as dst:
                shutil.copyfileobj(src, dst)

        if _sha256_of(FFMPEG_EXE) != FFMPEG_EXE_SHA256:
            print("  FFmpeg download failed verification (checksum")
            print("  mismatch) - deleting and continuing without it.")
            FFMPEG_EXE.unlink(missing_ok=True)
            return False

        print("  FFmpeg installed.")
        return True
    except Exception as e:
        print(f"  FFmpeg download failed: {e}")
        FFMPEG_EXE.unlink(missing_ok=True)
        return False
    finally:
        tmp_zip.unlink(missing_ok=True)


def ensure_ffmpeg():
    """Resolve a usable ffmpeg, downloading a private copy only if needed.

    Checks, in order: ffmpeg already on PATH (respected as-is, no download),
    then a previously-downloaded private copy, then downloads one. Never
    touches the system PATH - avoids the same registry/live-session desync
    bug diagnosed for HF_TOKEN, since a PATH edit would need a logoff/reboot
    to reliably take effect.

    Returns:
        The command string to use for ffmpeg ("ffmpeg" or a full path), or
        None if no ffmpeg is available and the download failed.
    """
    on_path = shutil.which("ffmpeg")
    if on_path:
        return "ffmpeg"

    if FFMPEG_EXE.exists():
        return str(FFMPEG_EXE)

    print()
    print("FFmpeg not found - needed for M4B export and speed adjustment.")
    if download_ffmpeg():
        return str(FFMPEG_EXE)

    print("  Continuing without FFmpeg - M4B export and speed-adjusted")
    print("  audio will not work until this succeeds on a future launch.")
    return None


def clear_screen():
    """Clear the console so the current step's instructions are always the
    only thing on screen (full reference stays open in Notepad)."""
    os.system("cls" if sys.platform == "win32" else "clear")


def hf_token_exists():
    """Return True if a HuggingFace token is already available via the
    HF_TOKEN environment variable or the standard token file."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        return HF_TOKEN_FILE.is_file() and HF_TOKEN_FILE.read_text().strip() != ""
    except OSError:
        # An unreadable token file is treated as no token.
        return False


def write_instructions_file():
    """Write the full HuggingFace setup instructions to a text file in the
    install folder and open it in the default text editor (Notepad), so the
    instructions stay visible even after they scroll off the console."""
    try:
        INSTRUCTIONS_FILE.write_text(INSTRUCTIONS_TEXT, encoding="utf-8")
        if sys.platform == "win32":
            os.startfile(str(INSTRUCTIONS_FILE))
    except Exception:
        # Setup must not die because Notepad could not open; the console
        # walkthrough repeats every step anyway.
        pass


def validate_hf_token(token):
    """Verify a pasted token against HuggingFace and pinpoint what is wrong.

    Args:
        token: The user-supplied access token string.

    Returns:
        One of: "ok" (token valid AND model terms accepted),
        "bad_token" (HuggingFace rejects the token itself),
        "terms_not_accepted" (token valid but gated terms not accepted),
        "gated_permission_missing" (terms accepted, but this fine-grained
        token lacks the separate "public gated repos" permission needed to
        actually download the file),
        "network_error" (could not reach HuggingFace).
    """
    import urllib.request
    import urllib.error

    def status_for(url, method="GET"):
        """Return the HTTP status HuggingFace gives this token for url.

        Args:
            url: The HuggingFace URL to check.
            method: HTTP method to use - "HEAD" for the large gated model
                file so we don't download hundreds of MB just to read a
                status code.
        """
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return None

    whoami = status_for("https://huggingface.co/api/whoami-v2")
    if whoami is None:
        return "network_error"
    if whoami != 200:
        return "bad_token"

    gated = status_for("https://huggingface.co/api/models/kyutai/pocket-tts")
    if gated is None:
        return "network_error"
    if gated != 200:
        # Valid account but the gated repo refuses it: terms not accepted yet.
        return "terms_not_accepted"

    # The metadata check above can pass even when the token can't actually
    # download the file - fine-grained tokens gate that behind a separate
    # permission. Probe the real download URL to catch it.
    file_check = status_for(GATED_FILE_CHECK_URL, method="HEAD")
    if file_check is None:
        return "network_error"
    if file_check == 200:
        return "ok"
    return "gated_permission_missing"


def save_hf_token(token):
    """Persist the token everywhere the app can pick it up.

    Writes it three ways: `setx` for a permanent user-level environment
    variable (future processes), os.environ for THIS process (so the very
    first model download works without a restart), and the standard
    huggingface_hub token file as a fallback.

    Args:
        token: The verified access token string.
    """
    os.environ["HF_TOKEN"] = token
    try:
        HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        HF_TOKEN_FILE.write_text(token)
    except OSError:
        print("  Warning: could not write the token file; using the")
        print("  environment variable only.")
    if sys.platform == "win32":
        r = subprocess.run(
            ["setx", "HF_TOKEN", token],
            capture_output=True, creationflags=NOCONSOLE
        )
        # setx failing is not fatal: the token file covers future runs.
        if r.returncode != 0:
            print("  Warning: could not set the HF_TOKEN environment")
            print("  variable; the saved token file will be used instead.")


def setup_hf_token():
    """Interactive walkthrough for gated-model access and token entry.

    Opens the full instructions in Notepad, then steps the user through:
    accepting the model terms (signing up first if needed), creating a
    fine-grained read token, and pasting it here. The token is verified
    live - a bad paste and un-accepted terms produce different, specific
    messages. Blank input skips; the app then runs with built-in voices
    only and a marker file stops the prompt repeating every launch.
    """
    import webbrowser

    write_instructions_file()

    # --- Step 1: model terms ---
    clear_screen()
    print("=" * 60)
    print("  Voice Cloning Setup - Step 1 of 2: Accept Model Terms")
    print("=" * 60)
    print()
    print("Pocket TTS clones voices using a GATED (but free) model on")
    print("HuggingFace. You must accept its terms once.")
    print()
    print("The full instructions are open in Notepad and saved as:")
    print(f"  {INSTRUCTIONS_FILE}")
    print()
    print("Your browser will now open the model page:")
    print(f"  {GATED_MODEL_URL}")
    print()
    print("If you do NOT have a HuggingFace account, the page will ask")
    print("you to sign up first - free, but required to use the model.")
    print("After signing up, verify your email and return to the model")
    print('page, then click "Agree and access repository".')
    print()
    input("Press Enter to open the model page in your browser... ")
    webbrowser.open(GATED_MODEL_URL)
    print()
    input('Press Enter AFTER you have clicked "Agree and access repository"... ')

    # --- Step 2: token ---
    clear_screen()
    print("=" * 60)
    print("  Voice Cloning Setup - Step 2 of 2: Access Token")
    print("=" * 60)
    print()
    print("Pocket TTS needs a HuggingFace access token (see Notepad for")
    print("the detailed version of these steps):")
    print()
    print('  1. On the tokens page, click "New token".')
    print('  2. Name it (e.g. "Personal Read Token").')
    print('  3. Token type: "Fine-grained".')
    print("  4. Under Repositories, check BOTH boxes:")
    print('     "Read access to contents of all public gated repos')
    print('      you can access" (REQUIRED - this is the one that')
    print("      actually enables voice cloning)")
    print('     "Read access to contents of all repos under your')
    print('      personal namespace" (optional)')
    print('  5. Click "Create token" and COPY it immediately -')
    print("     it is shown only once.")
    print()
    input("Press Enter to open the Access Tokens page in your browser... ")
    webbrowser.open(TOKEN_SETTINGS_URL)
    print()
    print("Paste your token below (right-click or Ctrl+V), then press")
    print("Enter. Press Enter on an empty line to SKIP - Pocket TTS will")
    print("still work with its built-in voices, but cloning your own")
    print("voice samples will be disabled.")
    print()

    for _ in range(3):
        token = input("Token: ").strip()

        if not token:
            # Explicit skip: remember it so the user is not nagged every
            # launch, and tell them how to come back.
            TOKEN_SKIP_MARKER.write_text("skipped")
            print()
            print("  Skipped. Voice cloning is disabled (built-in voices")
            print("  still work). To set it up later, delete this file and")
            print("  launch Pocket TTS again:")
            print(f"  {TOKEN_SKIP_MARKER}")
            print()
            input("Press Enter to continue... ")
            return

        print("  Checking your token with HuggingFace...")
        result = validate_hf_token(token)

        if result == "ok":
            save_hf_token(token)
            print()
            print("  SUCCESS - token verified and saved. Voice cloning is")
            print("  enabled.")
            print()
            input("Press Enter to continue... ")
            return

        if result == "terms_not_accepted":
            print()
            print("  Your token works, but the model terms have NOT been")
            print("  accepted yet. Reopening the model page - click")
            print('  "Agree and access repository", then paste the SAME')
            print("  token again.")
            print()
            webbrowser.open(GATED_MODEL_URL)
        elif result == "gated_permission_missing":
            print()
            print("  Your token is valid and the model terms are accepted,")
            print("  but this fine-grained token is missing a separate")
            print("  permission checkbox for gated repos.")
            print()
            print("  Go to the Access Tokens page (reopening it now), edit")
            print("  this token, and check the box:")
            print('    "Read access to contents of all public gated repos')
            print('     you can access"')
            print("  then save. The same token works once its permissions")
            print("  are updated - paste it again below.")
            print()
            webbrowser.open(TOKEN_SETTINGS_URL)
        elif result == "network_error":
            print()
            print("  Could not reach HuggingFace - check your internet")
            print("  connection and try again.")
            print()
        else:
            print()
            print("  That token was not recognized. Make sure you copied")
            print("  the whole thing (it starts with 'hf_') and paste it")
            print("  again.")
            print()

    print("  Three attempts failed - moving on. You will be asked again")
    print("  the next time you launch Pocket TTS.")
    print()
    input("Press Enter to continue... ")


def run_cmd(cmd, label=""):
    """Run a command with streaming output."""
    if label:
        print(f"  {label}...")
        sys.stdout.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        creationflags=NOCONSOLE if sys.platform == "win32" else 0
    )
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"    {line[:120]}")
    proc.wait()
    return proc


def run_setup():
    """Run first-time setup with console output."""
    print()
    print("=" * 60)
    print("  Pocket TTS First-Time Setup")
    print("=" * 60)
    print()
    print("This will download PyTorch, language models, and other")
    print("dependencies (~2-3 GB). It may take 10-20 minutes depending")
    print("on your internet speed.")
    print()

    print("[1/3] Upgrading pip...")
    r = run_cmd(
        [str(BUNDLED_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    if r.returncode != 0:
        print("  Warning: pip upgrade failed, continuing...")

    print("[2/3] Installing PyTorch (CUDA, ~2.8 GB)...")
    r = run_cmd(
        [str(BUNDLED_PYTHON), "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        "Installing PyTorch"
    )
    if r.returncode != 0:
        print("  CUDA install failed, falling back to CPU-only PyTorch...")
        r = run_cmd(
            [str(BUNDLED_PYTHON), "-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            "Installing PyTorch (CPU fallback)"
        )
        if r.returncode != 0:
            print("  FAILED: PyTorch installation failed. Check internet connection.")
            input("\nPress Enter to exit...")
            return False

    print("[3/3] Installing remaining dependencies...")
    if REQUIREMENTS_TXT.exists():
        r = run_cmd(
            [str(BUNDLED_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_TXT)],
            "Installing requirements"
        )
        if r.returncode != 0:
            print("  Warning: some requirements had issues, continuing...")
    else:
        print("  Warning: requirements.txt not found, skipping...")

    SETUP_MARKER.write_text("ok")
    print()
    print("=" * 60)
    print("  Setup complete! Launching Pocket TTS...")
    print("=" * 60)
    print()
    return True


# ASR uses the same Python env as the main app (no separate ASR/venv).
# faster-whisper / rapidfuzz / pywhispercpp are in requirements_windows.txt.


def main():
    if not LAUNCH_GUI.exists():
        print(f"ERROR: launch_gui.py not found at: {LAUNCH_GUI}")
        input("\nPress Enter to exit...")
        return 1

    if not BUNDLED_PYTHON.exists():
        print()
        print("=" * 60)
        print("  Pocket TTS Install Error")
        print("=" * 60)
        print()
        print("python.exe not found in the install directory.")
        print("Please reinstall Pocket TTS.")
        print()
        input("Press Enter to exit...")
        return 1

    if not SETUP_MARKER.exists():
        success = run_setup()
        if not success:
            return 1

    # Gated-model access: prompt whenever no token is available and the
    # user has not explicitly skipped - covers first run, an interrupted
    # setup, and a token that was later removed.
    if not hf_token_exists() and not TOKEN_SKIP_MARKER.exists():
        setup_hf_token()

    # Checked every launch (not just first run) so a deleted or corrupted
    # private copy is re-downloaded automatically.
    ffmpeg_cmd = ensure_ffmpeg()

# ASR packages live in the main env (root requirements.txt). No second venv.

    os.chdir(str(INSTALL_DIR))
    child_env = os.environ.copy()
    if ffmpeg_cmd is not None:
        child_env["POCKET_TTS_FFMPEG_PATH"] = ffmpeg_cmd
    subprocess.run(
        [str(BUNDLED_PYTHON), str(LAUNCH_GUI)],
        env=child_env
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
