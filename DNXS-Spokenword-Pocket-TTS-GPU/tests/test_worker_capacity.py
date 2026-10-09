"""Tests for runtime GPU worker capacity calculations."""

from pocket_tts.audiobook.generator import AudiobookGenerator, calculate_vram_worker_capacity


def test_capacity_uses_current_free_vram_and_reserve() -> None:
    """Capacity reflects free VRAM after reserving ten percent of total."""
    gib = 1024 ** 3

    capacity = calculate_vram_worker_capacity(
        free_bytes=int(6.5 * gib),
        total_bytes=8 * gib,
        worker_vram_gb=1.0,
        reserve_percent=10.0,
    )

    assert capacity == 5


def test_capacity_does_not_use_total_vram_as_available_vram() -> None:
    """Existing allocations reduce worker capacity before new workers spawn."""
    gib = 1024 ** 3

    capacity = calculate_vram_worker_capacity(
        free_bytes=int(3.3 * gib),
        total_bytes=8 * gib,
        worker_vram_gb=1.0,
        reserve_percent=10.0,
    )

    assert capacity == 2


def test_regeneration_worker_count_uses_stored_scheduler_plan() -> None:
    """Parallel recovery must read the plan stored on its generator instance."""
    generator = AudiobookGenerator.__new__(AudiobookGenerator)
    generator._regen_tts_workers = 1

    assert generator._regeneration_worker_count(3) == 1
