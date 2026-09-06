import os

import pytest

from v2.media.layout import LAYOUT_ENV_VAR, build_layout


def test_build_layout_creates_expected_directories(tmp_path):
    layout = build_layout("abc123", data_dir=tmp_path)

    assert layout.root == tmp_path / "abc123"
    assert layout.source_dir.is_dir()
    assert layout.audio_dir.is_dir()
    assert layout.proxy_dir.is_dir()
    assert layout.metadata_dir.is_dir()


def test_build_layout_is_idempotent(tmp_path):
    first = build_layout("abc123", data_dir=tmp_path)
    second = build_layout("abc123", data_dir=tmp_path)

    assert first == second


def test_build_layout_derives_data_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(LAYOUT_ENV_VAR, str(tmp_path / "env-root"))

    layout = build_layout("env-test")

    assert layout.root == tmp_path / "env-root" / "env-test"


def test_build_layout_default_data_dir_is_project_local(monkeypatch):
    monkeypatch.delenv(LAYOUT_ENV_VAR, raising=False)

    default = build_layout("default-test").root

    assert default.name == "default-test"
    assert "television-discourse-decoded-hindi" in str(default)


def test_layout_paths_are_deterministic(tmp_path):
    layout = build_layout("vid", data_dir=tmp_path)

    assert layout.original_video == tmp_path / "vid" / "source" / "original.mp4"
    assert layout.audio_path == tmp_path / "vid" / "audio" / "audio.wav"
    assert layout.proxy_path == tmp_path / "vid" / "proxy" / "proxy_480p.mp4"
    assert layout.metadata_path == tmp_path / "vid" / "metadata" / "metadata.json"
    assert layout.asr_path == tmp_path / "vid" / "asr" / "transcription.json"


def test_build_layout_creates_asr_dir(tmp_path):
    layout = build_layout("vid", data_dir=tmp_path)

    assert layout.asr_dir.is_dir()