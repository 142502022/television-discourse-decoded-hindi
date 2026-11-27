#!/usr/bin/env python3
"""Fast, auditable topic-alignment and topic-drift analysis.

Inputs:
  data/v2/debate_only/videos/<VIDEO_ID>/<VARIANT>/merge_final.json
  data/v2/debate_only/overall_topic/<VARIANT>/<VIDEO_ID>/overall_topic.json
  data/v2/debate_only/subtopic_analysis/<VARIANT>/<VIDEO_ID>/subtopic_analysis.json

If Stage 3 contains meaningful subtopics, score each subtopic. If Stage 3 has only
one generic/empty subtopic (as in the supplied sample), fall back to transcript
windows so the episode can still be analysed without waiting for Stage 3 repair.

Requires:
  pip install sentence-transformers torch

Scores use equal weighting of overall-topic cosine similarity and the mean of
the two strongest key-point similarities. The display index is clipped cosine
similarity multiplied by 100; it is NOT a percentage or calibrated probability.
Thresholds are provisional and require manual validation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "v2" / "debate_only" / "videos"
DEFAULT_OVERALL = ROOT / "data" / "v2" / "debate_only" / "overall_topic"
DEFAULT_SUBTOPIC = ROOT / "data" / "v2" / "debate_only" / "subtopic_analysis"
DEFAULT_OUTPUT = ROOT / "data" / "v2" / "debate_only" / "topic_drift"
VARIANTS = ("baseline", "threshold_0.5", "threshold_1.0", "threshold_2.0", "threshold_5.0")
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    raw = data.get("segments")
    if not isinstance(raw, list):
        raise ValueError(f"No 'segments' array in {path}")
    rows = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = clean_text(s.get("text"))
        if start >= 0 and end > start and text:
            rows.append({"start": start, "end": end, "text": text,
                         "speaker_id": s.get("speaker") or s.get("speaker_id")})
    rows.sort(key=lambda x: x["start"])
    if not rows:
        raise ValueError(f"No usable transcript segments in {path}")
    return rows


def make_windows(segments: list[dict[str, Any]], width: float, stride: float):
    duration = max(s["end"] for s in segments)
    windows = []
    start = 0.0
    idx = 1
    while start < duration:
        end = min(start + width, duration)
        members = [s for s in segments if s["start"] < end and s["end"] > start]
        if members:
            text = " ".join(s["text"] for s in members)
            windows.append({
                "window_id": f"TW{idx:03d}",
                "start": round(start, 3), "end": round(end, 3),
                "text": text,
                "speakers": sorted({str(s["speaker_id"]) for s in members if s["speaker_id"]}),
                "segment_count": len(members)
            })
            idx += 1
        start += stride
    return windows


def meaningful_subtopics(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("subtopics")
    if not isinstance(rows, list):
        return []
    good = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title"))
        desc = clean_text(row.get("description"))
        generic = title.lower() in {"opening discussion", "discussion segment", "untitled subtopic", ""}
        if generic and not desc:
            continue
        if not desc and generic:
            continue
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        evidence = row.get("evidence", [])
        evidence_text = " ".join(clean_text(x.get("text")) for x in evidence if isinstance(x, dict))
        content = " ".join(x for x in [title, desc, evidence_text] if x)
        good.append({
            "unit_id": row.get("subtopic_id", f"ST{i+1:03d}"),
            "title": title or f"Subtopic {i+1}",
            "start": start, "end": end,
            "text": content,
            "source": "stage3_subtopic"
        })
    return good


def classify(score: float, low: float, high: float) -> str:
    if score >= high:
        return "aligned"
    if score >= low:
        return "related_or_partial_drift"
    return "possible_topic_drift"


def find_drift_runs(rows: list[dict[str, Any]], min_duration: float) -> list[dict[str, Any]]:
    """Merge adjacent low-alignment windows; require a minimum sustained duration."""
    runs = []
    active = None
    for row in rows:
        low = row["classification"] == "possible_topic_drift"
        if low:
            if active is None:
                active = {"start": row["start"], "end": row["end"], "scores": [row["alignment_score"]], "ids": [row["unit_id"]]}
            else:
                active["end"] = max(active["end"], row["end"])
                active["scores"].append(row["alignment_score"])
                active["ids"].append(row["unit_id"])
        elif active is not None:
            if active["end"] - active["start"] >= min_duration:
                runs.append(active)
            active = None
    if active is not None and active["end"] - active["start"] >= min_duration:
        runs.append(active)
    return [{
        "start": round(r["start"], 3), "end": round(r["end"], 3),
        "duration_seconds": round(r["end"] - r["start"], 3),
        "mean_alignment_score": round(sum(r["scores"]) / len(r["scores"]), 2),
        "window_ids": r["ids"],
        "interpretation": "Sustained low semantic alignment; review transcript context before concluding drift."
    } for r in runs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--overall-topic-dir", type=Path, default=DEFAULT_OVERALL)
    ap.add_argument("--subtopic-dir", type=Path, default=DEFAULT_SUBTOPIC)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    ap.add_argument("--window-seconds", type=float, default=60.0)
    ap.add_argument("--stride-seconds", type=float, default=30.0)
    ap.add_argument("--low-threshold", type=float, default=20.0)
    ap.add_argument("--high-threshold", type=float, default=40.0)
    ap.add_argument("--min-drift-seconds", type=float, default=90.0)
    args = ap.parse_args()

    transcript_path = args.data_dir / args.video_id / args.variant / "merge_final.json"
    overall_path = args.overall_topic_dir / args.variant / args.video_id / "overall_topic.json"
    subtopic_path = args.subtopic_dir / args.variant / args.video_id / "subtopic_analysis.json"
    for path in (transcript_path, overall_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    overall_data = read_json(overall_path)
    overall = overall_data.get("overall_topic", {})
    if not isinstance(overall, dict):
        raise ValueError("overall_topic.json is missing its overall_topic object")
    topic = clean_text(overall.get("topic"))
    description = clean_text(overall.get("description"))
    key_points = overall.get("key_discussion_points", [])
    key_points = [clean_text(x) for x in key_points if clean_text(x)] if isinstance(key_points, list) else []
    if not topic:
        raise ValueError("Overall topic title is empty")

    segments = load_segments(transcript_path)
    stage3 = read_json(subtopic_path) if subtopic_path.is_file() else {}
    units = meaningful_subtopics(stage3)
    source_mode = "subtopic_level"
    if not units:
        source_mode = "transcript_windows_fallback_stage3_not_informative"
        units = make_windows(segments, args.window_seconds, args.stride_seconds)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install sentence-transformers torch") from exc

    model = SentenceTransformer(args.embedding_model)
    # Title + description + key points represent the central topic. Keeping key
    # points separate allows a subtopic to align through a related issue/angle.
    topic_text = " ".join([topic, description])
    reference_texts = [topic_text] + key_points
    ref_vecs = model.encode(reference_texts, normalize_embeddings=True, show_progress_bar=False)
    unit_vecs = model.encode([u["text"] for u in units], normalize_embeddings=True, show_progress_bar=False)

    results = []
    for unit, vec in zip(units, unit_vecs):
        sims = (ref_vecs @ vec).tolist()
        title_sim = float(sims[0])

        # Compare the subtopic with every key discussion point, then average
        # the two strongest matches. If only one point exists, use that one.
        point_sims = [float(x) for x in sims[1:]]
        ranked_points = sorted(
            enumerate(point_sims, start=1), key=lambda item: item[1], reverse=True
        )
        top_matches = ranked_points[:2]
        if top_matches:
            top2_keypoint_similarity = sum(score for _, score in top_matches) / len(top_matches)
        else:
            top2_keypoint_similarity = title_sim

        # Equal weighting: half broad-topic similarity, half top-two key-point similarity.
        combined_cosine = 0.50 * title_sim + 0.50 * top2_keypoint_similarity

        # Do NOT use (cosine + 1) * 50: that adds an artificial 50-point offset.
        # This 0-100 display index is clipped cosine * 100, NOT a percentage.
        alignment_index = max(0.0, min(100.0, combined_cosine * 100.0))
        row = {
            "unit_id": unit["unit_id"],
            "unit_type": "subtopic" if source_mode == "subtopic_level" else "time_window",
            "title": unit.get("title"),
            "start": round(unit["start"], 3), "end": round(unit["end"], 3),
            "duration_seconds": round(unit["end"] - unit["start"], 3),
            "semantic_similarity_to_overall_topic": round(title_sim, 4),
            "key_point_similarities": [
                {
                    "key_point": key_points[index - 1],
                    "cosine_similarity": round(score, 4)
                }
                for index, score in top_matches
            ],
            "top_two_key_point_similarity": round(top2_keypoint_similarity, 4),
            "weighted_cosine_alignment": round(combined_cosine, 4),
            "alignment_score": round(alignment_index, 2),
            "classification": classify(alignment_index, args.low_threshold, args.high_threshold),
            "speakers": unit.get("speakers", []),
            "evidence_excerpt": unit["text"][:1200]
        }
        results.append(row)

    # Drift runs are generated from chronological time windows. For actual Stage 3
    # subtopics, a low-scoring subtopic is flagged individually; no claim of precise
    # onset/return is made unless the fallback windows are available.
    if source_mode == "subtopic_level":
        drift_events = [{
            "start": r["start"], "end": r["end"],
            "duration_seconds": r["duration_seconds"],
            "mean_alignment_score": r["alignment_score"],
            "unit_ids": [r["unit_id"]],
            "interpretation": "Subtopic has low semantic alignment; inspect its transcript evidence."
        } for r in results if r["classification"] == "possible_topic_drift"]
    else:
        drift_events = find_drift_runs(results, args.min_drift_seconds)

    output = {
        "schema_version": "topic-drift-1.0",
        "video_id": args.video_id,
        "variant": args.variant,
        "overall_topic": {"topic": topic, "description": description, "key_discussion_points": key_points},
        "method": {
            "name": "multilingual_embedding_alignment",
            "embedding_model": args.embedding_model,
            "analysis_mode": source_mode,
            "score_formula": "0.50 * cosine(unit, overall_topic+description) + 0.50 * mean(two highest cosine similarities between unit and individual key discussion points)",
            "display_score": "alignment_score = max(0, weighted_cosine_alignment) * 100; this is a scaled cosine index, NOT a percentage or probability",
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "threshold_status": "provisional; calibrate against manually labelled subtopics before research claims",
            "minimum_sustained_drift_seconds": args.min_drift_seconds,
            "warning": "Scores are heuristic semantic-alignment indicators, not probabilities or validated accuracy. Low similarity is a candidate for review, not proof of drift. Thresholds are provisional."
        },
        "alignment_results": results,
        "drift_events": drift_events,
        "summary": {
            "units_analysed": len(results),
            "aligned_units": sum(r["classification"] == "aligned" for r in results),
            "partially_aligned_units": sum(r["classification"] == "related_or_partial_drift" for r in results),
            "possible_drift_units": sum(r["classification"] == "possible_topic_drift" for r in results),
            "drift_events": len(drift_events),
            "analysis_mode": source_mode
        }
    }
    out = args.output_dir / args.variant / args.video_id / "topic_drift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
