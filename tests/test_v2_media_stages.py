"""Milestone 1 stage tests, using a local FFmpeg sample (no network)."""

from pathlib import Path

import pytest

from v2.media.audio import extract_audio
from v2.media.download import find_original_video
from v2.media.metadata import curate_metadata, save_metadata
from v2.media.proxy import create_proxy

from conftest import audio_stream, ffprobe_streams, video_stream


def test_find_original_video_none_when_empty(tmp_path):
    assert find_original_video(tmp_path) is None


def test_find_original_video_finds_non_empty_file(tmp_path):
    existing = tmp_path / "original.mp4"
    existing.write_bytes(b"\x00" * 32)

    assert find_original_video(tmp_path) == existing


def test_extract_audio_produces_16k_mono_wav(sample_video, tmp_path):
    output = extract_audio(sample_video, tmp_path / "audio.wav")

    assert output.exists() and output.stat().st_size > 0
    stream = audio_stream(output)
    assert stream["codec_name"] == "pcm_s16le"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1
    assert stream["sample_fmt"] == "s16"


def test_extract_audio_is_idempotent(sample_video, tmp_path):
    output_path = tmp_path / "audio.wav"
    first = extract_audio(sample_video, output_path)
    second = extract_audio(sample_video, output_path)

    assert first == second


def test_create_proxy_is_mp4_h264_with_height_capped(sample_video, tmp_path):
    output = create_proxy(sample_video, tmp_path / "proxy_480p.mp4")

    assert output.exists() and output.stat().st_size > 0

    stream = video_stream(output)
    assert stream["codec_name"] == "h264"
    assert int(stream["width"]) % 2 == 0
    assert int(stream["height"]) <= 480

    probe = ffprobe_streams(output)
    assert "mp4" in probe["format"]["format_name"].split(",")
    assert audio_stream(output)["codec_name"] == "aac"


def test_create_proxy_is_idempotent(sample_video, tmp_path):
    output_path = tmp_path / "proxy_480p.mp4"
    first = create_proxy(sample_video, output_path)
    second = create_proxy(sample_video, output_path)

    assert first == second


def test_curate_metadata_keeps_only_expected_fields():
    full = {
        "id": "vid123",
        "title": "Big Debate",
        "duration": 600,
        "upload_date": "20260101",
        "requested_downloads": [{"filepath": "/tmp/original.mp4"}],
        "formats": [{"format_id": "137"}],
    }

    curated = curate_metadata(full)

    assert curated["id"] == "vid123"
    assert curated["title"] == "Big Debate"
    assert curated["duration"] == 600
    assert "requested_downloads" not in curated
    assert "formats" not in curated
    assert "fetched_at" in curated


def test_save_metadata_roundtrips_curated_fields(tmp_path):
    output = save_metadata(curate_metadata({"id": "x", "title": "T"}), tmp_path / "m.json")

    import json

    with output.open("r", encoding="utf-8") as file_obj:
        saved = json.load(file_obj)
    assert saved["id"] == "x"
    assert saved["title"] == "T"


def test_save_metadata_replaces_corrupt_existing_file(tmp_path):
    output_path = tmp_path / "m.json"
    output_path.write_text('{"id": "truncated"', encoding="utf-8")

    save_metadata({"id": "replacement"}, output_path)

    import json

    with output_path.open("r", encoding="utf-8") as file_obj:
        saved = json.load(file_obj)
    assert saved["id"] == "replacement"