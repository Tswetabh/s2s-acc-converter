"""CLI tool to display system hardware & environment info."""
import sys
import os
import subprocess

# Auto-redirect to project virtual environment on Drive D if executed with global Python
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"))
if os.path.exists(venv_python):
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(os.path.abspath(venv_python))
    if curr_exe != target_exe:
        result = subprocess.run([venv_python] + sys.argv, check=False)
        sys.exit(result.returncode)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.system_info import print_system_banner

if __name__ == "__main__":
    print_system_banner()
