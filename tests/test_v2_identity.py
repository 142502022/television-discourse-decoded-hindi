"""Tests for the role-inference heuristic (pure logic, no model)."""

from v2.identity.roles import infer_roles, segment_is_question


def _segment(start, end, text, speaker, words=None):
    seg = {"start": start, "end": end, "text": text, "speaker": speaker}
    if words is not None:
        seg["words"] = words
    return seg


def test_segment_is_question():
    assert segment_is_question({"text": "How are they coping?"})
    assert segment_is_question({"text": "What is happening on the ground"})
    assert not segment_is_question({"text": "Absolutely, that is true"})


def test_anchor_is_first_longest_questioner():
    segments = [
        _segment(0, 5, "Good evening, joining us is Geeta, what are the updates?", "SPK_A"),
        _segment(5, 7, "Thank you, here is what we know", "SPK_B"),
        _segment(7, 9, "And how are people coping?", "SPK_A"),
        _segment(9, 11, "They are anxious, supplies are short", "SPK_B"),
    ]
    roles = infer_roles(segments)
    assert roles["anchor_speaker"] == "SPK_A"
    assert roles["roles"]["SPK_A"]["role"] == "anchor"
    assert roles["roles"]["SPK_B"]["role"] == "participant"
    assert "evidence" in roles["roles"]["SPK_A"]


def test_interruption_direction_influences_role():
    segments = [
        _segment(0, 6, "Welcome, tell us what you saw", "SPK_A"),
        _segment(6, 12, "We saw heavy flooding overnight", "SPK_B"),
    ]
    interruptions = [
        {"interrupter": "SPK_A", "victim": "SPK_B", "seconds": 1.0, "count": 6},
    ]
    roles = infer_roles(segments, interruptions)
    assert roles["anchor_speaker"] == "SPK_A"


def test_namer_replaces_speaker_labels():
    segments = [
        _segment(0, 4, "Hello there", "SPK_A"),
        _segment(4, 8, "Hi how are you", "SPK_B"),
    ]
    roles = infer_roles(
        segments,
        namer=lambda s: {"SPK_A": "Anchor", "SPK_B": "Guest"}.get(s, s),
    )
    assert roles["anchor"] == "Anchor"
    assert roles["roles"]["SPK_B"]["name"] == "Guest"