"""Tests for the pure speaker-assignment logic (no pyannote import)."""

from v2.speech.diarize import (
    assign_speakers_to_segments,
    build_diarized_transcript,
    dominant_speaker,
)

TURNS = [
    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
    {"start": 3.0, "end": 7.0, "speaker": "SPEAKER_01"},
    {"start": 8.0, "end": 9.0, "speaker": "SPEAKER_02"},
]


def test_dominant_speaker_by_coverage():
    assert dominant_speaker(0.0, 3.0, TURNS) == "SPEAKER_00"
    assert dominant_speaker(3.0, 7.0, TURNS) == "SPEAKER_01"
    assert dominant_speaker(3.5, 4.5, TURNS) == "SPEAKER_01"


def test_dominant_speaker_returns_none_when_no_overlap():
    assert dominant_speaker(9.5, 10.0, TURNS) is None


def test_assign_tags_segments_and_words():
    segments = [
        {
            "start": 0.5,
            "end": 2.0,
            "text": "Hello",
            "words": [{"word": "Hello", "start": 0.5, "end": 2.0}],
        },
        {"start": 4.0, "end": 9.0, "text": "Crosstalk"},
    ]
    tagged = assign_speakers_to_segments(segments, TURNS)

    assert tagged[0]["speaker"] == "SPEAKER_00"
    assert tagged[0]["words"][0]["speaker"] == "SPEAKER_00"
    assert tagged[1]["speaker"] == "SPEAKER_01"
    assert tagged is not segments
    assert tagged[0] is not segments[0]


def test_build_diarized_transcript_is_self_contained():
    transcript = {
        "model": "medium",
        "language": "en",
        "aligned": True,
        "segments": [{"start": 1.0, "end": 2.0, "text": "Hi"}],
    }
    meta = {"model": "pyannote/speaker-diarization-3.1", "device": "mps", "num_speakers": 2}
    merged = build_diarized_transcript(transcript, TURNS, meta)

    assert merged["diarization"] == meta
    assert merged["speaker_turns"] == TURNS
    assert merged["segments"][0]["speaker"] == "SPEAKER_00"
    assert merged["model"] == "medium"