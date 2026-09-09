"""Tests for step 11 (chunking) + step 12 (Label Studio export)."""

from v2.chunk.chunking import chunk_plan, slice_events, slice_segments
from v2.labelstudio.export import build_label_studio_tasks, build_task, write_tasks_jsonl


# ── Step 11: chunking ────────────────────────────────────────────────────

def test_chunk_plan_covers_duration():
    plan = chunk_plan(1900, chunk_seconds=900)
    assert [c["index"] for c in plan] == [0, 1, 2]
    assert plan[0] == {"index": 0, "start": 0, "end": 900}
    assert plan[1] == {"index": 1, "start": 900, "end": 1800}
    assert plan[2] == {"index": 2, "start": 1800, "end": 1900}


def test_chunk_plan_exact_multiple_single_chunk():
    plan = chunk_plan(450, chunk_seconds=900)
    assert len(plan) == 1
    assert plan[0]["end"] == 450


def test_slice_segments_clips_to_window():
    segs = [
        {"start": 10, "end": 100, "text": "a"},
        {"start": 200, "end": 300, "text": "b"},
    ]
    sliced = slice_segments(segs, 50, 210)
    assert len(sliced) == 2  # both overlap [50, 210)
    assert sliced[0]["start"] == 50
    assert sliced[0]["end"] == 100
    assert sliced[1]["start"] == 200
    assert sliced[1]["end"] == 210


def test_slice_segments_excludes_outside():
    segs = [{"start": 10, "end": 30, "text": "a"}]
    assert slice_segments(segs, 50, 210) == []


def test_slice_events_same_boundary_rule():
    events = [
        {"start": 10, "end": 20},
        {"start": 25, "end": 30},
    ]
    sliced = slice_events(events, 15, 26)
    assert [e["start"] for e in sliced] == [15, 25]


# ── Step 12: Label Studio export ─────────────────────────────────────────

def test_build_task_one_task_per_chunk():
    payload = {
        "chunk_index": 0,
        "start_s": 0,
        "end_s": 900,
        "video": "/abs/path/chunk_000.mp4",
        "segments": [{"start": 0, "end": 5}],
        "overlap_candidates": [{"start": 1, "end": 2}],
    }
    task = build_task(payload, "VIDEO0")
    assert task["data"]["episode_id"] == "VIDEO0"
    assert task["data"]["chunk_index"] == 0
    assert task["data"]["video"] == "/abs/path/chunk_000.mp4"
    assert task["annotations"] == []


def test_build_task_uses_base_url_for_relative_video():
    payload = {
        "chunk_index": 1,
        "start_s": 900,
        "end_s": 1000,
        "video": "/abs/chunks/chunk_001.mp4",
        "segments": [],
        "overlap_candidates": [],
    }
    task = build_task(payload, "VIDEO1", base_url="http://localhost:8080/media")
    assert task["data"]["video"] == "http://localhost:8080/media/chunk_001.mp4"


def test_build_label_studio_tasks_sorted_by_chunk():
    chunks = [
        {"chunk_index": 2, "video": "c2.mp4", "segments": [], "overlap_candidates": []},
        {"chunk_index": 0, "video": "c0.mp4", "segments": [], "overlap_candidates": []},
        {"chunk_index": 1, "video": "c1.mp4", "segments": [], "overlap_candidates": []},
    ]
    tasks = build_label_studio_tasks(chunks, "EP")
    assert [t["data"]["chunk_index"] for t in tasks] == [0, 1, 2]


def test_write_tasks_jsonl_roundtrip(tmp_path):
    tasks = [
        {"data": {"video": "c0.mp4", "episode_id": "E", "chunk_index": 0},
         "annotations": []},
        {"data": {"video": "c1.mp4", "episode_id": "E", "chunk_index": 1},
         "annotations": []},
    ]
    out = write_tasks_jsonl(tasks, tmp_path / "tasks.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json as _json
    assert _json.loads(lines[1])["data"]["chunk_index"] == 1