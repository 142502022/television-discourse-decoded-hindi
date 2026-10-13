"""Tests for per-video ASR language override resolution."""

import json

from v2.cli.transcribe import load_language_overrides, resolve_language


def test_explicit_language_wins_over_override():
    assert resolve_language("vid", "en", {"vid": "hi"}) == "en"


def test_override_applied_when_no_explicit_language():
    assert resolve_language("vid", None, {"vid": "en"}) == "en"


def test_falls_back_to_none_for_auto_detection():
    assert resolve_language("vid", None, {"other": "en"}) is None


def test_load_overrides_reads_mapping_and_ignores_bad_files(tmp_path):
    good = tmp_path / "overrides.json"
    good.write_text(json.dumps({"a": "en", "b": "hi", "c": 5, "d": ""}), encoding="utf-8")
    assert load_language_overrides(good) == {"a": "en", "b": "hi"}

    broken = tmp_path / "broken.json"
    broken.write_text("{broken", encoding="utf-8")
    assert load_language_overrides(broken) == {}
    assert load_language_overrides(tmp_path / "missing.json") == {}
    assert load_language_overrides(None) == {}
