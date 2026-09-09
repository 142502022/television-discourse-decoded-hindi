"""Tests for step-7 roster prompt/parse helpers (pure, no model import)."""

from v2.roster.extract import (
    build_prompt,
    build_roster_artifact,
    normalize_roster,
    parse_roster,
    system_prompt,
)

SEGMENTS = [
    {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_01", "text": "We begin with our anchor, R. Menon."},
    {"start": 3.6, "end": 9.2, "speaker": "SPEAKER_00", "text": "Thank you for having me."},
]


def test_build_prompt_includes_speaker_tagged_turns():
    prompt = build_prompt(SEGMENTS)
    assert "[0.0-3.5] SPEAKER_01: We begin with our anchor, R. Menon." in prompt
    assert "[3.6-9.2] SPEAKER_00: Thank you for having me." in prompt
    assert "SPEAKER_01" in prompt


def test_system_prompt_mentions_schema_and_json_only():
    prompt = system_prompt()
    assert "Return ONLY valid JSON" in prompt
    assert "gender_as_addressed" in prompt
    assert "evidence_timestamps_s" in prompt


def test_parse_roster_strips_markdown_fences():
    raw = '```json\n{"participants": [{"name": "R. Menon"}]}\n```'
    payload = parse_roster(raw)
    assert payload["participants"] == [{"name": "R. Menon"}]


def test_parse_roster_tolerates_surrounding_prose():
    raw = 'Preamble here. {"participants": [{"name": "A"}]} trailing prose'
    assert parse_roster(raw)["participants"] == [{"name": "A"}]


def test_parse_roster_rejects_missing_participants():
    try:
        parse_roster('{"foo": true}')
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_normalize_roster_fills_defaults_and_marks_unknown():
    cleaned = normalize_roster(
        [
            {
                "name": "  Geeta Mohan ",
                "speaker_label": "SPEAKER_00",
                "role": "panelist",
                "affiliation": None,
                "gender_as_addressed": "female",
                "evidence_timestamps_s": [1.2, "3.4"],
                "evidence_quote": "Geeta reports",
            },
            {"name": "", "role": "bogus"},
        ]
    )
    assert cleaned[0]["name"] == "Geeta Mohan"
    assert cleaned[0]["evidence_timestamps_s"] == [1.2, 3.4]
    assert cleaned[0]["affiliation"] is None
    assert cleaned[1]["name"] == "Unknown"
    assert cleaned[1]["role"] == "unknown"
    assert cleaned[1]["gender_as_addressed"] == "unknown"
    assert cleaned[1]["speaker_label"] == ""
    assert cleaned[0]["speaker_label"] == "SPEAKER_00"


def test_build_roster_artifact_is_uniform():
    artifact = build_roster_artifact(
        [
            {"name": "R. Menon", "speaker_label": "SPEAKER_01", "role": "anchor"},
        ],
        "gemini-2.5-flash",
    )
    assert artifact["method"] == "llm-extraction"
    assert artifact["model"] == "gemini-2.5-flash"
    assert artifact["num_participants"] == 1
    assert artifact["participants"][0]["name"] == "R. Menon"