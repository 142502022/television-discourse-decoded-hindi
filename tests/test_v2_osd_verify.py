"""Tests for step-6 reconciliation logic (pure, no model imports)."""

from v2.osd.verify import (
    CONFIDENCE_WEIGHTS,
    build_verification,
    classify_candidates,
    covered_by_osd,
    speaker_intersections,
)

TURNS = [
    {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
    {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
    {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_00"},
    {"start": 13.0, "end": 14.0, "speaker": "SPEAKER_01"},
]


def test_speaker_intersections_find_different_speaker_overlaps():
    intervals = speaker_intersections(TURNS)
    assert intervals == [
        {"start": 4.0, "end": 6.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
        {"start": 13.0, "end": 14.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
    ]


def test_covered_by_osd_detects_corroborated_interval():
    interval = {"start": 4.0, "end": 6.0}
    assert covered_by_osd(interval, [{"start": 4.0, "end": 6.0}]) is True
    assert covered_by_osd(interval, [{"start": 0.0, "end": 1.0}]) is False


def test_osd_with_two_speakers_is_agree():
    results = classify_candidates([{"start": 4.0, "end": 6.0}], TURNS)
    agree = [r for r in results if r["case"] == "AGREE"]
    assert len(agree) == 1
    assert agree[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert agree[0]["confidence"] == CONFIDENCE_WEIGHTS["AGREE"] == 1.0


def test_osd_with_one_speaker_is_osd_only():
    results = classify_candidates([{"start": 0.5, "end": 1.5}], TURNS)
    osd_only = [r for r in results if r["case"] == "OSD_ONLY"]
    assert len(osd_only) == 1
    assert osd_only[0]["confidence"] == CONFIDENCE_WEIGHTS["OSD_ONLY"] == 0.7


def test_turn_intersection_without_osd_is_diarization_only():
    # SPEAKER_00/SPEAKER_01 overlap at [13,14) but no OSD region fires there.
    results = classify_candidates([{"start": 4.0, "end": 6.0}], TURNS)
    cases = [r["case"] for r in results]
    assert cases == ["AGREE", "DIARIZATION_ONLY"]
    d_only = next(r for r in results if r["case"] == "DIARIZATION_ONLY")
    assert d_only["start"] == 13.0
    assert d_only["end"] == 14.0
    assert d_only["confidence"] == CONFIDENCE_WEIGHTS["DIARIZATION_ONLY"] == 0.6


def test_corroborated_intersection_is_not_duplicated():
    # Both turn intersections lie inside OSD regions -> no DIARIZATION_ONLY.
    regions = [{"start": 3.5, "end": 6.5}, {"start": 12.5, "end": 14.5}]
    results = classify_candidates(regions, TURNS)
    assert all(r["case"] == "AGREE" for r in results)
    assert len(results) == 2


def test_full_verification_artifact_summary():
    verification = build_verification([{"start": 4.0, "end": 6.0}], TURNS)
    summary = verification["summary"]
    assert summary["num_agree"] == 1
    assert summary["num_osd_only"] == 0
    assert summary["num_diarization_only"] == 1
    assert summary["num_candidates"] == 2
    assert verification["method"] == "three-case-reconciliation"
    assert sorted(verification["cases"]) == sorted(CONFIDENCE_WEIGHTS)