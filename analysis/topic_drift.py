#!/usr/bin/env python3
"""Stage: subtopic-to-overall-topic alignment and sustained topic-drift analysis.

Variant restricted to threshold_5.0. Uses the existing subtopic boundaries and
full transcript segments, plus the overall-topic key discussion points.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

VARIANT = "threshold_5.0"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOW_THRESHOLD = 20.0
DEFAULT_HIGH_THRESHOLD = 40.0
DEFAULT_MIN_SUSTAINED_SECONDS = 90.0


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def get_speaker(seg: dict[str, Any]) -> Optional[str]:
    value = seg.get("speaker_id", seg.get("speaker"))
    return str(value) if value is not None and str(value).strip() else None


def transcript_for_interval(segments: list[dict[str, Any]], start: float, end: float) -> tuple[str, list[str]]:
    """Collect segment text whose midpoint falls inside [start, end)."""
    selected: list[tuple[float, str, Optional[str]]] = []
    for seg in segments:
        try:
            ss, se = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        midpoint = (ss + se) / 2.0
        if start <= midpoint < end:
            text = clean_text(seg.get("text"))
            if text:
                selected.append((ss, text, get_speaker(seg)))
    selected.sort(key=lambda x: x[0])
    speakers = sorted({sp for _, _, sp in selected if sp})
    return clean_text(" ".join(text for _, text, _ in selected)), speakers


def build_unit_text(subtopic: dict[str, Any], transcript_segments: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    start, end = float(subtopic["start"]), float(subtopic["end"])
    transcript_text, transcript_speakers = transcript_for_interval(transcript_segments, start, end)
    title = clean_text(subtopic.get("title"))
    description = clean_text(subtopic.get("description"))
    evidence_texts = [clean_text(e.get("text")) for e in subtopic.get("evidence", []) if isinstance(e, dict)]
    # Prefer full transcript text. Evidence is a fallback only, avoiding duplicate inflation.
    body = transcript_text or clean_text(" ".join(evidence_texts))
    unit_text = clean_text(". ".join(x for x in (title, description, body) if x))
    speakers = transcript_speakers or sorted({
        str(e.get("speaker_id")) for e in subtopic.get("evidence", [])
        if isinstance(e, dict) and e.get("speaker_id") is not None
    })
    source = "merge_final.segments" if transcript_text else ("subtopic_analysis.evidence_fallback" if body else "metadata_only")
    return unit_text, speakers, source


def classify(score: float, low: float, high: float) -> str:
    if score < low:
        return "possible_drift"
    if score < high:
        return "partially_aligned"
    return "aligned"


def make_drift_events(results: list[dict[str, Any]], min_duration: float, max_gap: float = 30.0) -> list[dict[str, Any]]:
    """Group consecutive low-alignment subtopics; require sustained duration."""
    candidates = [r for r in results if r["classification"] == "possible_drift"]
    groups: list[list[dict[str, Any]]] = []
    for row in candidates:
        if not groups or row["start"] - groups[-1][-1]["end"] > max_gap:
            groups.append([row])
        else:
            groups[-1].append(row)
    events = []
    for group in groups:
        start, end = group[0]["start"], group[-1]["end"]
        duration = max(0.0, end - start)
        if duration < min_duration:
            continue
        events.append({
            "drift_event_id": f"DRIFT_{len(events)+1:03d}",
            "start": round(start, 3), "end": round(end, 3),
            "duration_seconds": round(duration, 3),
            "subtopic_ids": [x["unit_id"] for x in group],
            "minimum_alignment_score": min(x["alignment_score"] for x in group),
            "mean_alignment_score": round(sum(x["alignment_score"] for x in group) / len(group), 2),
            "interpretation": "Sustained low semantic alignment; candidate topic drift requiring contextual/manual review."
        })
    return events


def analyse(video_id: str, variant: str, overall_data: dict[str, Any], subtopic_data: dict[str, Any],
            merge_data: dict[str, Any], model_name: str, low: float, high: float,
            min_sustained: float) -> dict[str, Any]:
    if variant != VARIANT:
        raise ValueError(f"Only {VARIANT} is supported")
    for label, obj in (("overall-topic", overall_data), ("subtopic", subtopic_data)):
        found_id = obj.get("video_id")
        if found_id not in (None, video_id):
            raise ValueError(f"{label} video_id mismatch: expected {video_id}, got {found_id}")
        found_variant = obj.get("variant")
        if found_variant not in (None, variant):
            raise ValueError(f"{label} variant mismatch: expected {variant}, got {found_variant}")

    overall = overall_data.get("overall_topic", {})
    topic = clean_text(overall.get("topic"))
    description = clean_text(overall.get("description"))
    key_points = [clean_text(x) for x in overall.get("key_discussion_points", []) if clean_text(x)]
    if not topic:
        raise ValueError("overall_topic.topic is missing")
    if not key_points:
        raise ValueError("overall_topic.key_discussion_points is empty; cannot calculate key-point alignment")

    subtopics = subtopic_data.get("subtopics", [])
    if not isinstance(subtopics, list) or not subtopics:
        raise ValueError("subtopic_analysis.subtopics is missing or empty")
    segments = merge_data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("merge_final.segments must be an array")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Install: pip install sentence-transformers scikit-learn numpy") from exc

    model = SentenceTransformer(model_name)
    topic_text = clean_text(f"{topic}. {description}")
    unit_records = []
    texts_to_encode = [topic_text] + key_points
    for st in subtopics:
        try:
            start, end = float(st["start"]), float(st["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid start/end for subtopic: {st.get('subtopic_id', '<unknown>')}")
        if end <= start:
            raise ValueError(f"Subtopic end must be after start: {st.get('subtopic_id')}")
        unit_text, speakers, source = build_unit_text(st, segments)
        if not unit_text:
            raise ValueError(f"No text available for subtopic {st.get('subtopic_id')}")
        unit_records.append((st, start, end, unit_text, speakers, source))
        texts_to_encode.append(unit_text)

    embeddings = model.encode(texts_to_encode, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    topic_vec = embeddings[0]
    key_vecs = embeddings[1:1 + len(key_points)]
    results = []
    for idx, (st, start, end, unit_text, speakers, source) in enumerate(unit_records):
        unit_vec = embeddings[1 + len(key_points) + idx]
        topic_sim = float(np.dot(unit_vec, topic_vec))
        point_sims = sorted(
            [{"key_point": kp, "cosine_similarity": float(np.dot(unit_vec, kv))}
             for kp, kv in zip(key_points, key_vecs)],
            key=lambda x: x["cosine_similarity"], reverse=True
        )
        top_two = point_sims[:2]
        top_two_mean = sum(x["cosine_similarity"] for x in top_two) / len(top_two)
        weighted = 0.50 * topic_sim + 0.50 * top_two_mean
        score = max(0.0, weighted) * 100.0
        results.append({
            "unit_id": st.get("subtopic_id", f"ST{idx+1:03d}"),
            "unit_type": "subtopic",
            "title": st.get("title"),
            "start": round(start, 3), "end": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "semantic_similarity_to_overall_topic": round(topic_sim, 4),
            "key_point_similarities": [{"key_point": x["key_point"], "cosine_similarity": round(x["cosine_similarity"], 4)} for x in point_sims],
            "top_two_key_point_similarity": round(top_two_mean, 4),
            "weighted_cosine_alignment": round(weighted, 4),
            "alignment_score": round(score, 2),
            "classification": classify(score, low, high),
            "speakers": speakers,
            "text_source": source,
            "evidence_excerpt": unit_text[:1800]
        })

    results.sort(key=lambda x: (x["start"], x["end"]))
    drift_events = make_drift_events(results, min_sustained)
    counts = Counter(x["classification"] for x in results)
    return {
        "schema_version": "topic-drift-1.1",
        "video_id": video_id,
        "variant": variant,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_topic": {"topic": topic, "description": description, "key_discussion_points": key_points},
        "method": {
            "name": "subtopic_to_overall_topic_embedding_alignment",
            "embedding_model": model_name,
            "analysis_mode": "subtopic_level",
            "unit_text_source": "Full transcript segments within each subtopic interval; subtopic evidence is fallback only.",
            "score_formula": "0.50 * cosine(unit, overall_topic+description) + 0.50 * mean(two highest cosine similarities between unit and individual key discussion points)",
            "display_score": "alignment_score = max(0, weighted_cosine_alignment) * 100; scaled cosine index, not a percentage or probability",
            "low_threshold": low,
            "high_threshold": high,
            "threshold_interpretation": f"possible_drift < {low}; partially_aligned >= {low} and < {high}; aligned >= {high}",
            "minimum_sustained_drift_seconds": min_sustained,
            "sustained_drift_rule": "Consecutive possible_drift subtopics separated by no more than 30 seconds are grouped; event duration must meet the minimum sustained duration.",
            "threshold_status": "provisional; manually calibrate before research claims",
            "warning": "Scores are heuristic semantic-alignment indicators, not probabilities or validated accuracy. Low similarity is a candidate for review, not proof of drift."
        },
        "alignment_results": results,
        "drift_events": drift_events,
        "summary": {
            "units_analysed": len(results),
            "aligned_units": counts.get("aligned", 0),
            "partially_aligned_units": counts.get("partially_aligned", 0),
            "possible_drift_units": counts.get("possible_drift", 0),
            "drift_events": len(drift_events),
            "analysis_mode": "subtopic_level"
        }
    }


def process_one(root: Path, video_id: str, args: argparse.Namespace) -> Path:
    base = root / "data/v2/debate_only"
    merge_path = base / "videos" / video_id / VARIANT / "merge_final.json"
    overall_path = base / "overall_topic" / VARIANT / video_id / "overall_topic.json"
    subtopic_path = base / "subtopic_analysis" / VARIANT / video_id / "subtopic_analysis.json"
    for path in (merge_path, overall_path, subtopic_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    result = analyse(video_id, args.variant, read_json(overall_path), read_json(subtopic_path),
                     read_json(merge_path), args.embedding_model, args.low_threshold,
                     args.high_threshold, args.minimum_sustained_drift_seconds)
    output_path = base / "topic_drift" / VARIANT / video_id / "topic_drift.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Subtopic alignment and topic drift (threshold_5.0 only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--low-threshold", type=float, default=DEFAULT_LOW_THRESHOLD)
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    parser.add_argument("--minimum-sustained-drift-seconds", type=float, default=DEFAULT_MIN_SUSTAINED_SECONDS)
    args = parser.parse_args()
    if args.low_threshold >= args.high_threshold:
        parser.error("--low-threshold must be lower than --high-threshold")
    if args.minimum_sustained_drift_seconds < 0:
        parser.error("--minimum-sustained-drift-seconds cannot be negative")
    root = args.root.resolve()
    videos_dir = root / "data/v2/debate_only/videos"
    video_ids = [args.video_id] if args.video_id else sorted(
        p.name for p in videos_dir.iterdir() if p.is_dir() and (p / VARIANT / "merge_final.json").exists()
    ) if videos_dir.exists() else []
    if not video_ids:
        print(f"No {VARIANT} videos found.")
        return 1
    ok = skipped = failed = 0
    for video_id in video_ids:
        print(f"\nTopic drift: {video_id} / {VARIANT}", flush=True)
        try:
            out = process_one(root, video_id, args)
            print(f"[OK] Wrote {out}", flush=True)
            ok += 1
        except FileNotFoundError as exc:
            print(f"[SKIP] {video_id}: {exc}", file=sys.stderr)
            skipped += 1
        except Exception as exc:
            print(f"[FAIL] {video_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    print(f"\nSummary: videos={len(video_ids)}, successful={ok}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())