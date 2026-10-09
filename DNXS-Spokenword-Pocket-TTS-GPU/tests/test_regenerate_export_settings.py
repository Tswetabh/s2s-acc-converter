"""Regression coverage for Regenerate-tab Main export setting handoff."""

from types import SimpleNamespace

from pocket_tts.gui.regenerate_tab import RegenerateTab


def test_main_export_settings_uses_live_main_tab_values():
    """Overlay current Main-tab controls on persistent export defaults."""
    main_window = SimpleNamespace(
        config=SimpleNamespace(
            m4b={"sample_rate": 24000, "target_db": -1.5, "normalization_type": "peak"}
        ),
        _m4b_gui_settings=lambda: {
            "write_m4b": True,
            "write_mp3": True,
            "write_wav": False,
            "normalization_type": "loudness",
            "chapterize": True,
            "chapter_mode": "headings_or_minutes",
            "max_chapter_minutes": 45,
        },
    )
    tab = SimpleNamespace(window=lambda: main_window)

    settings = RegenerateTab._main_export_settings(tab)

    assert settings["write_m4b"] is True
    assert settings["write_mp3"] is True
    assert settings["write_wav"] is False
    assert settings["normalization_type"] == "loudness"
    assert settings["chapterize"] is True
    assert settings["chapter_mode"] == "headings_or_minutes"
    assert settings["max_chapter_minutes"] == 45
    assert settings["sample_rate"] == 24000
    assert settings["target_db"] == -1.5
