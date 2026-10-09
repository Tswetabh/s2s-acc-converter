"""GPU and RAM memory monitoring tools for TTS benchmarking."""

import os
import gc
import psutil
from typing import Dict, Any, Optional

class MemorySnapshot:
    """Snapshot of VRAM and RAM at an instant."""
    def __init__(self):
        self.vram_used_gb = 0.0
        self.vram_free_gb = 0.0
        self.ram_used_gb = 0.0
        self.update()

    def update(self):
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                self.vram_free_gb = round(free_b / (1024 ** 3), 3)
                self.vram_used_gb = round((total_b - free_b) / (1024 ** 3), 3)
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()
            self.ram_used_gb = round(vm.used / (1024 ** 3), 3)
        except Exception:
            pass

class GPUMonitor:
    """Tracks memory delta and peak consumption during TTS execution."""

    @staticmethod
    def cleanup():
        """Aggressively reclaim GPU and system memory."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def reset_peak_stats():
        """Reset PyTorch peak memory trackers."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass

    @staticmethod
    def get_peak_vram_gb(baseline_used_gb: float) -> float:
        """Calculate peak VRAM reached during operation."""
        try:
            import torch
            if torch.cuda.is_available():
                torch_peak_b = torch.cuda.max_memory_allocated(0)
                torch_peak_gb = torch_peak_b / (1024 ** 3)
                # Global device VRAM check
                free_b, total_b = torch.cuda.mem_get_info(0)
                current_used_gb = (total_b - free_b) / (1024 ** 3)
                return round(max(baseline_used_gb + torch_peak_gb, current_used_gb), 3)
        except Exception:
            pass
        return baseline_used_gb

    @staticmethod
    def capture_state() -> Dict[str, float]:
        """Capture current memory metrics."""
        snap = MemorySnapshot()
        return {
            "vram_used_gb": snap.vram_used_gb,
            "vram_free_gb": snap.vram_free_gb,
            "ram_used_gb": snap.ram_used_gb,
        }
