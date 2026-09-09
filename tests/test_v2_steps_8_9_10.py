"""Tests for step 9 name linking + step 10 finalize + step 8 sampler."""

from v2.identity.link import link_roster
from v2.validation.sampler import build_sample_artifact, compute_stats, sample_timestamps

ROSTER_PARTICIPANTS = [
    {"name": "Geeta Mohan", "speaker_label": "SPEAKER_00", "role": "correspondent",
     "affiliation": "India Today", "gender_as_addressed": "female",
     "evidence_timestamps_s": [1.2, 3.4], "evidence_quote": "Geeta reports"},
    {"name": "Pooja", "speaker_label": "SPEAKER_01", "role": "anchor",
     "affiliation": "India Today", "gender_as_addressed": "female",
     "evidence_timestamps_s": [0.1], "evidence_quote": "Pooja: hello"},
]

ROLES_DATA = {
    "method": "discourse-heuristic",
    "anchor": "SPEAKER_01",
    "anchor_speaker": "SPEAKER_01",
    "total_speech_s": 540.0,
    "roles": {
        "SPEAKER_00": {"role": "participant", "score": 3.5, "confidence": 0.4,
                        "name": "SPEAKER_00",
                        "evidence": {"first_start_s": 17.0, "speech_seconds": 332.0,
                                     "speech_share": 0.677, "segment_count": 77,
                                     "questions": 4, "addresses": 8,
                                     "interruptions_initiated": 4, "interruptions_received": 27}},
        "SPEAKER_01": {"role": "anchor", "score": 5.4, "confidence": 0.6,
                        "name": "SPEAKER_01",
                        "evidence": {"first_start_s": 0.1, "speech_seconds": 158.0,
                                     "speech_share": 0.323, "segment_count": 44,
                                     "questions": 21, "addresses": 6,
                                     "interruptions_initiated": 27, "interruptions_received": 4}},
    },
}

POOL_VIDEOS = [
    {"video_id": "aaa", "segments": [
        {"start": 0.0, "end": 5.0, "speaker": "S0", "text": "hello world"},
        {"start": 6.0, "end": 9.0, "speaker": "S1", "text": "goodbye"},
    ]},
    {"video_id": "bbb", "segments": [
        {"start": 1.0, "end": 3.0, "speaker": "S0", "text": "test phrase"},
    ]},
]


# ── Step 9: link_roster ──────────────────────────────────────────────────

def test_link_roster_merges_roster_and_roles():
    linked = link_roster(ROSTER_PARTICIPANTS, ROLES_DATA)
    assert linked["anchor_label"] == "SPEAKER_01"
    assert linked["anchor_name"] == "Pooja"
    assert linked["num_participants"] == 2
    by_label = {p["speaker_label"]: p for p in linked["participants"]}
    assert by_label["SPEAKER_00"]["name"] == "Geeta Mohan"
    assert by_label["SPEAKER_00"]["role"] == "participant"
    assert by_label["SPEAKER_00"]["speech_seconds"] == 332.0
    assert by_label["SPEAKER_01"]["interruptions_initiated"] == 27


def test_link_roster_handles_missing_roster():
    linked = link_roster([], ROLES_DATA)
    assert linked["num_participants"] == 0
    assert linked["participants"] == []


def test_link_roster_handles_missing_roles():
    linked = link_roster(ROSTER_PARTICIPANTS, None)
    by_label = {p["speaker_label"]: p for p in linked["participants"]}
    assert by_label["SPEAKER_00"]["name"] == "Geeta Mohan"
    assert by_label["SPEAKER_00"]["role"] == "correspondent"
    assert by_label["SPEAKER_00"]["speech_seconds"] is None


# ── Step 8: sampler ──────────────────────────────────────────────────────

def test_sample_timestamps_deterministic_with_seed():
    samples = sample_timestamps(POOL_VIDEOS, n=2, seed=42)
    assert len(samples) == 2
    assert all("video_id" in s for s in samples)
    assert all("timestamp_s" in s for s in samples)
    assert all("speaker_label" in s for s in samples)
    assert all("expected_text" in s for s in samples)


def test_sample_timestamps_n_clamped_to_available():
    samples = sample_timestamps(POOL_VIDEOS, n=100, seed=42)
    assert len(samples) == 3  # only 3 segments total


def test_build_sample_artifact_wraps_samples():
    samples = sample_timestamps(POOL_VIDEOS, n=2, seed=1)
    art = build_sample_artifact(samples, model="whisperx-medium")
    assert art["method"] == "random-evidence-sample"
    assert art["n_samples"] == 2


def test_compute_stats_full_pass():
    samples = [
        {"annulment": "pass"}, {"annulment": "pass"}, {"annulment": "fail"}
    ]
    stats = compute_stats(samples)
    assert stats["total"] == 3
    assert stats["pass"] == 2
    assert stats["fail"] == 1
    assert stats["accuracy"] == round(2 / 3, 4)
    assert stats["error_rate"] == round(1 / 3, 4)


def test_compute_stats_unanswered():
    stats = compute_stats([{"annulment": None}, {"annulment": None}])
    assert stats["unanswered"] == 2
    assert stats["accuracy"] is None
