"""Tests for the transcript payload builder (no torch/whisperx import)."""

from v2.speech.transcribe import build_transcript_payload, default_compute_type


def test_default_compute_type_per_device():
    assert default_compute_type("cpu") == "int8"
    assert default_compute_type("mps") == "float16"
    assert default_compute_type("cuda") == "float16"


def test_payload_maps_segments_and_words():
    raw = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "  Hello world  ",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.7},
                    {"word": "world", "start": 0.8, "end": 2.5},
                ],
            }
        ],
    }

    payload = build_transcript_payload(
        raw,
        model="medium",
        device="mps",
        compute_type="float16",
        aligned=True,
    )

    assert payload["model"] == "medium"
    assert payload["language"] == "en"
    assert payload["aligned"] is True
    assert payload["segments"][0]["text"] == "Hello world"
    assert payload["segments"][0]["words"][1] == {"word": "world", "start": 0.8, "end": 2.5}


def test_payload_drops_words_when_not_aligned():
    raw = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}]}

    payload = build_transcript_payload(
        raw,
        model="base",
        device="cpu",
        compute_type="int8",
        aligned=False,
    )

    assert payload["aligned"] is False
    assert "words" not in payload["segments"][0]