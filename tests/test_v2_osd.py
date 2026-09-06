"""Tests for the pure OSD analysis logic (no pyannote import)."""

from v2.osd.overlap import (
    build_analysis,
    interrupter_of,
    interruption_stats,
    overlap_events,
    pair_stats,
)

TURNS = [
    {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
    {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
    {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_00"},
    {"start": 13.0, "end": 14.0, "speaker": "SPEAKER_01"},
]


def test_interrupter_detects_floor_holder_vs_interloper():
    region = {"start": 4.0, "end": 5.0}
    assert interrupter_of(region, TURNS) == "SPEAKER_01"


def test_interrupter_none_when_no_ongoing_speaker():
    region = {"start": 11.0, "end": 12.5}  # gap between turns, no floor holder
    assert interrupter_of(region, TURNS) is None


def test_overlap_events_attribute_speakers_and_duration():
    events = overlap_events([{"start": 4.0, "end": 6.0}], TURNS)
    event = events[0]
    assert event["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert event["interrupter"] == "SPEAKER_01"
    assert event["duration"] == 2.0


def test_pair_stats_aggregate_per_unordered_pair():
    stats = pair_stats(overlap_events([{"start": 4.0, "end": 6.0}, {"start": 5.0, "end": 5.5}], TURNS))
    assert stats == [{"pair": ["SPEAKER_00", "SPEAKER_01"], "seconds": 2.5, "count": 2}]


def test_interruption_stats_are_directed():
    events = [
        {
            "start": 4.0,
            "end": 5.0,
            "duration": 1.0,
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "interrupter": "SPEAKER_01",
        }
    ]
    assert interruption_stats(events) == [
        {"interrupter": "SPEAKER_01", "victim": "SPEAKER_00", "seconds": 1.0, "count": 1}
    ]


def test_interruption_stats_keep_directions_separate():
    events = [
        {
            "start": 4.0,
            "end": 5.0,
            "duration": 1.0,
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "interrupter": "SPEAKER_01",
        },
        {
            "start": 10.0,
            "end": 11.0,
            "duration": 1.0,
            "speakers": ["SPEAKER_01", "SPEAKER_00"],
            "interrupter": "SPEAKER_00",
        },
    ]
    assert interruption_stats(events) == [
        {"interrupter": "SPEAKER_00", "victim": "SPEAKER_01", "seconds": 1.0, "count": 1},
        {"interrupter": "SPEAKER_01", "victim": "SPEAKER_00", "seconds": 1.0, "count": 1},
    ]


def test_build_analysis_summarizes():
    analysis = build_analysis(
        [{"start": 4.0, "end": 6.0}],
        TURNS,
        {"model": "pyannote/overlap-detection", "device": "mps"},
    )
    assert analysis["detector"]["model"] == "pyannote/overlap-detection"
    assert analysis["stats"]["num_regions"] == 1
    assert analysis["stats"]["overlap_seconds"] == 2.0
    assert analysis["stats"]["num_interruptions"] == 1
    assert analysis["stats"]["total_duration_s"] == 15.0
    assert analysis["interruptions"][0]["interrupter"] == "SPEAKER_01"