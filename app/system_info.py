"""System hardware and runtime detection for TTS POC Evaluation Platform."""

import os
import platform
import sys
import subprocess
from typing import Dict, Any, Optional

def get_cpu_info() -> str:
    """Retrieve CPU brand string."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output("wmic cpu get name", shell=True, text=True, stderr=subprocess.DEVNULL)
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
    except Exception:
        pass
    return platform.processor() or "AMD/Intel CPU"

def get_ram_info() -> Dict[str, float]:
    """Retrieve total and available system RAM in GB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "used_gb": round(vm.used / (1024 ** 3), 2)
        }
    except Exception:
        # Fallback for Windows without psutil yet
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return {
                "total_gb": round(stat.ullTotalPhys / (1024 ** 3), 2),
                "available_gb": round(stat.ullAvailPhys / (1024 ** 3), 2),
                "used_gb": round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3), 2)
            }
        except Exception:
            return {"total_gb": 16.0, "available_gb": 10.0, "used_gb": 6.0}

def get_gpu_info() -> Dict[str, Any]:
    """Retrieve NVIDIA GPU, VRAM, and driver details."""
    gpu_data = {
        "available": False,
        "name": "N/A",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "used_vram_gb": 0.0,
        "cuda_version": "N/A",
        "driver_version": "N/A"
    }
    
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            gpu_data["available"] = True
            gpu_data["name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            gpu_data["total_vram_gb"] = round(total_b / (1024 ** 3), 2)
            gpu_data["free_vram_gb"] = round(free_b / (1024 ** 3), 2)
            gpu_data["used_vram_gb"] = round((total_b - free_b) / (1024 ** 3), 2)
            gpu_data["cuda_version"] = torch.version.cuda or "N/A"
            return gpu_data
    except Exception:
        pass

    # Try nvidia-smi fallback
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 5:
            gpu_data["available"] = True
            gpu_data["name"] = parts[0]
            gpu_data["total_vram_gb"] = round(float(parts[1]) / 1024.0, 2)
            gpu_data["free_vram_gb"] = round(float(parts[2]) / 1024.0, 2)
            gpu_data["used_vram_gb"] = round(float(parts[3]) / 1024.0, 2)
            gpu_data["driver_version"] = parts[4]
            # Try getting CUDA version
            try:
                smi_out = subprocess.check_output("nvidia-smi", shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in smi_out.splitlines():
                    if "CUDA Version:" in line:
                        gpu_data["cuda_version"] = line.split("CUDA Version:")[1].split()[0]
            except Exception:
                pass
    except Exception:
        pass

    return gpu_data

def get_system_environment() -> Dict[str, Any]:
    """Compile comprehensive runtime & hardware metadata."""
    ram = get_ram_info()
    gpu = get_gpu_info()
    
    torch_version = "Not Installed"
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        pass

    os_name = f"{platform.system()} {platform.release()}"
    if platform.system() == "Windows" and sys.getwindowsversion().build >= 22000:
        os_name = "Windows 11"

    return {
        "os": os_name,
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "ram_total_gb": ram["total_gb"],
        "ram_available_gb": ram["available_gb"],
        "ram_used_gb": ram["used_gb"],
        "gpu_available": gpu["available"],
        "gpu_name": gpu["name"],
        "vram_total_gb": gpu["total_vram_gb"],
        "vram_free_gb": gpu["free_vram_gb"],
        "vram_used_gb": gpu["used_vram_gb"],
        "cuda_version": gpu["cuda_version"],
        "torch_version": torch_version,
        "driver_version": gpu["driver_version"]
    }

def print_system_banner():
    """Print standard formatted system banner."""
    env = get_system_environment()
    print("=" * 45)
    print("SYSTEM")
    print("-" * 45)
    print(f"{'GPU':<15} {env['gpu_name']}")
    print(f"{'VRAM':<15} {env['vram_total_gb']} GB")
    print(f"{'Free VRAM':<15} {env['vram_free_gb']} GB")
    print(f"{'RAM':<15} {env['ram_total_gb']} GB (Free: {env['ram_available_gb']} GB)")
    print(f"{'CPU':<15} {env['cpu']}")
    print(f"{'CUDA':<15} {env['cuda_version']}")
    print(f"{'PyTorch':<15} {env['torch_version']}")
    print(f"{'Python':<15} {env['python_version']}")
    print(f"{'OS':<15} {env['os']}")
    print("=" * 45)
    if env["vram_total_gb"] > 0 and env["vram_total_gb"] <= 6.0:
        print("[NOTICE] Target machine is 6.0 GB VRAM constrained (RTX 3050 Laptop).")
        print("         Models >1B parameters or requiring >5GB VRAM will be classified CLOUD_RECOMMENDED.")
        print("-" * 45)

if __name__ == "__main__":
    print_system_banner()
