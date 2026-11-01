#!/usr/bin/env python3
"""
Aggregate MTP per-video analyses into final_result.json.

- Processes all folders under data/v2/videos by default.
- Optional --video-id for single-episode testing.
- Never edits source JSON files.
- Missing/incomplete analyses are represented with explicit statuses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SOURCE_FILES = {
    "pipeline": "final.json",
    "overlap_followup": "overlap_followup_analysis.json",
    "overlap_followup_classified": "overlap_followup_classified.json",
    "turn_gap": "turn_gap_analysis.json",
    "overlap_clusters": "overlap_cluster_analysis.json",
    "floor_management": "floor_management_analysis.json",
    "host_actions": "host_actions.json",
    "topic_modeling": "topic_modeling.json",
    "host_identification": "host_identification.json",
    "host_action": "host_action.json",
    "interruption_management": "interruption_management.json",
    "overlap_duration": "overlap_duration_analysis.json",
}
OUTPUT_NAME = "final_result.json"


# ----------------------------- I/O and helpers -----------------------------

def load_json(path: Path) -> Tuple[Optional[Any], str]:
    if not path.exists():
        return None, "not_available"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), "available"
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] {path}: {exc}")
        return {"_read_error": str(exc)}, "error"


def get_path(obj: Any, *keys: str, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_value(obj: Dict[str, Any], *keys, default=None):
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return default


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def list_of_dicts(value) -> List[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def entropy_from_counts(counts: Counter) -> Optional[float]:
    total = sum(counts.values())
    if total == 0:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def source_status(source, status: str) -> str:
    if status != "available":
        return status
    if isinstance(source, dict) and "_read_error" in source:
        return "error"
    return "available"


def wrapped(source, status, description):
    return {"status": source_status(source, status), "description": description, "data": source}


# ----------------------------- Pipeline extraction -----------------------------

def extract_turns(pipeline: Optional[dict]) -> List[dict]:
    turns = get_path(pipeline, "speaker_turns", default=[])
    result = []
    for t in list_of_dicts(turns):
        speaker = first_value(t, "speaker", "speaker_id", "speaker_label")
        start = safe_float(first_value(t, "start", "start_time"))
        end = safe_float(first_value(t, "end", "end_time"))
        if speaker is not None and start is not None and end is not None and end >= start:
            result.append({"speaker": str(speaker), "start": start, "end": end, "duration": end - start, "source": t})
    return sorted(result, key=lambda x: (x["start"], x["end"]))


def extract_overlap_candidates(pipeline: Optional[dict]) -> List[dict]:
    candidates = get_path(pipeline, "overlap_candidates", default=[])
    return list_of_dicts(candidates)


def episode_duration(pipeline: Optional[dict]) -> Optional[float]:
    return safe_float(first_value(get_path(pipeline, "episode", default={}) or {},
                                  "duration", "duration_seconds"))


def metadata(pipeline: Optional[dict], video_id: str) -> dict:
    episode = get_path(pipeline, "episode", default={}) or {}
    meta = get_path(pipeline, "metadata", default={}) or {}
    return {
        "video_id": video_id,
        "title": first_value(episode, "title", default=meta.get("title")),
        "channel": first_value(episode, "channel", default=meta.get("channel")),
        "url": first_value(episode, "url", default=meta.get("url")) or
                f"https://www.youtube.com/watch?v={video_id}",
        "published_at": first_value(episode, "published_at", default=meta.get("published_at")),
        "duration_seconds": episode_duration(pipeline),
        "language": first_value(episode, "language", default=meta.get("language")),
    }


def overview(pipeline: Optional[dict]) -> dict:
    if pipeline is None:
        return {"status": "not_available"}
    turns = extract_turns(pipeline)
    asr_segments = get_path(pipeline, "asr", "segments")
    osd_regions = get_path(pipeline, "osd", "overlap_regions")
    speakers = sorted({t["speaker"] for t in turns})
    return {
        "status": "available",
        "duration_seconds": episode_duration(pipeline),
        "speaker_count": len(speakers) if turns else None,
        "speaker_turn_count": len(turns) if isinstance(get_path(pipeline, "speaker_turns"), list) else None,
        "asr_segment_count": len(asr_segments) if isinstance(asr_segments, list) else None,
        "overlap_candidate_count": len(extract_overlap_candidates(pipeline)) if isinstance(get_path(pipeline, "overlap_candidates"), list) else None,
        "osd_overlap_region_count": len(osd_regions) if isinstance(osd_regions, list) else None,
    }


# ----------------------------- Speaker participation -----------------------------

def analyze_participation(pipeline: Optional[dict], turn_gap: Optional[dict], duration: Optional[float]) -> dict:
    turns = extract_turns(pipeline)
    if not turns:
        return {"status": "not_available", "speakers": None}

    by_speaker = defaultdict(list)
    for t in turns:
        by_speaker[t["speaker"]].append(t)

    total_speech = sum(t["duration"] for t in turns)
    result = {}
    for speaker, rows in by_speaker.items():
        durations = [t["duration"] for t in rows]
        speaker_speech = sum(durations)
        result[speaker] = {
            "turn_count": len(rows),
            "total_speaking_time_seconds": round(speaker_speech, 3),
            "speech_share_percent": round(100 * speaker_speech / total_speech, 3) if total_speech else None,
            "mean_turn_duration_seconds": round(mean(durations), 3) if durations else None,
            "median_turn_duration_seconds": round(median(durations), 3) if durations else None,
            "longest_turn_duration_seconds": round(max(durations), 3) if durations else None,
            "first_speech_start": min(t["start"] for t in rows),
            "last_speech_end": max(t["end"] for t in rows),
        }

    # Floor-holding metrics are preserved from the dedicated turn-gap source;
    # schema-specific values are not guessed.
    floor_data = {
        "status": "available" if turn_gap is not None else "not_available",
        "source": turn_gap,
        "note": "Floor-holding intervals and internal gaps are retained from turn_gap_analysis; values are not re-derived without its exact schema.",
    }
    return {
        "status": "computed",
        "total_speech_time_seconds": round(total_speech, 3),
        "speaker_count": len(result),
        "speakers": dict(sorted(result.items(), key=lambda kv: kv[1]["total_speaking_time_seconds"], reverse=True)),
        "floor_holding": floor_data,
    }


# ----------------------------- Speaker transitions -----------------------------

def analyze_transitions(pipeline: Optional[dict], duration: Optional[float]) -> dict:
    turns = extract_turns(pipeline)
    if len(turns) < 2:
        return {"status": "not_available", "total_transitions": None}

    transitions = []
    pair_counts = Counter()
    outgoing = Counter()
    incoming = Counter()
    from_counts = Counter()
    short_turn_count = 0

    for prev, curr in zip(turns, turns[1:]):
        if prev["speaker"] == curr["speaker"]:
            continue
        gap = curr["start"] - prev["end"]
        rec = {
            "from_speaker": prev["speaker"],
            "to_speaker": curr["speaker"],
            "timestamp": curr["start"],
            "gap_seconds": round(gap, 3),
            "is_overlapping_transition": gap < 0,
            "previous_turn_duration_seconds": round(prev["duration"], 3),
        }
        transitions.append(rec)
        pair_counts[(prev["speaker"], curr["speaker"])] += 1
        outgoing[prev["speaker"]] += 1
        incoming[curr["speaker"]] += 1
        from_counts[prev["speaker"]] += 1
        if prev["duration"] < 1.0:
            short_turn_count += 1

    total = len(transitions)
    minutes = duration / 60 if duration and duration > 0 else None
    speakers = sorted(set(outgoing) | set(incoming))
    matrix = {s: {t: pair_counts[(s, t)] for t in speakers} for s in speakers}
    return {
        "status": "computed",
        "total_transitions": total,
        "transitions_per_minute": round(total / minutes, 4) if minutes else None,
        "short_turn_count": short_turn_count,
        "short_turn_frequency_percent": round(100 * short_turn_count / len(turns), 3) if turns else None,
        "transition_entropy_bits": round(entropy_from_counts(from_counts), 4),
        "pairwise_directional_counts": [
            {"from_speaker": a, "to_speaker": b, "count": n}
            for (a, b), n in sorted(pair_counts.items())
        ],
        "transition_matrix": matrix,
        "outgoing_transitions": dict(outgoing),
        "incoming_transitions": dict(incoming),
        "transition_events": transitions,
        "note": "These are structural speaker changes, not automatically interruptions.",
    }


# ----------------------------- Overlap and cluster integration -----------------------------

def overlap_stats(pipeline: Optional[dict], cluster_data: Optional[dict], duration: Optional[float]) -> dict:
    candidates = extract_overlap_candidates(pipeline)
    if pipeline is None or not isinstance(get_path(pipeline, "overlap_candidates"), list):
        return {"status": "not_available"}
    durations = [safe_float(c.get("duration")) for c in candidates]
    durations = [d for d in durations if d is not None and d >= 0]
    total_duration = sum(durations)
    minutes = duration / 60 if duration and duration > 0 else None

    bins = {
        "very_short_0_to_0_2s": sum(0 <= d <= 0.2 for d in durations),
        "short_over_0_2_to_0_5s": sum(0.2 < d <= 0.5 for d in durations),
        "moderate_over_0_5_to_1s": sum(0.5 < d <= 1.0 for d in durations),
        "prolonged_over_1s": sum(d > 1.0 for d in durations),
    }
    # Cluster classification remains the authoritative B-G source.
    cluster_summary = get_path(cluster_data, "statistics", "overlap_interaction_clusters", default=None)
    return {
        "status": "computed",
        "overlap_event_count": len(candidates),
        "total_overlap_duration_seconds": round(total_duration, 3),
        "overlap_events_per_minute": round(len(candidates) / minutes, 4) if minutes else None,
        "mean_overlap_duration_seconds": round(mean(durations), 3) if durations else None,
        "median_overlap_duration_seconds": round(median(durations), 3) if durations else None,
        "maximum_overlap_duration_seconds": round(max(durations), 3) if durations else None,
        "minimum_overlap_duration_seconds": round(min(durations), 3) if durations else None,
        "duration_bins_count": bins,
        "overlap_ratio_percent": round(100 * total_duration / duration, 3) if duration and duration > 0 else None,
        "source_counts": dict(Counter(str(c.get("source", "unknown")) for c in candidates)),
        "cluster_B_to_G_source_summary": cluster_summary,
        "raw_overlap_candidates": candidates,
        "note": "Duration bins are descriptive. B/C require conversational-function evidence; short duration alone does not assign a semantic cluster.",
    }


# ----------------------------- Main aggregation -----------------------------

def build_final_result(video_dir: Path) -> dict:
    video_id = video_dir.name
    loaded = {}
    statuses = {}
    for key, filename in SOURCE_FILES.items():
        loaded[key], statuses[key] = load_json(video_dir / filename)

    pipeline = loaded.get("pipeline")
    turn_gap = loaded.get("turn_gap")
    cluster_data = loaded.get("overlap_clusters")
    followup = loaded.get("overlap_followup")
    classified_followup = loaded.get("overlap_followup_classified")
    topic = loaded.get("topic_modeling")
    duration = episode_duration(pipeline)

    host_identification = loaded.get("host_identification")
    host_actions = loaded.get("host_actions") or loaded.get("host_action")

    return {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_metadata": metadata(pipeline, video_id),
        "episode_overview": overview(pipeline),

        "speaker_participation": analyze_participation(pipeline, turn_gap, duration),
        "speaker_transitions": analyze_transitions(pipeline, duration),

        "overlap_analysis": {
            "summary": overlap_stats(pipeline, cluster_data, duration),
            "followup_analysis": wrapped(followup, statuses["overlap_followup"],
                "Post-overlap speech observations in the configured follow-up window."),
            "followup_classification": wrapped(classified_followup, statuses["overlap_followup_classified"],
                "Conversational-function labels for post-overlap utterances."),
            "source_pipeline": {
                "status": statuses["pipeline"],
                "overlap_candidates": get_path(pipeline, "overlap_candidates"),
                "osd_overlap_regions": get_path(pipeline, "osd", "overlap_regions"),
                "verification": get_path(pipeline, "verify"),
            },
        },

        "interruption_analysis": wrapped(
            loaded.get("interruption_management"), statuses["interruption_management"],
            "Interruption management analysis. Success/failure counts are not inferred if absent from source."
        ),

        "interaction_clusters": wrapped(
            cluster_data, statuses["overlap_clusters"],
            "Cluster A structural transition candidates; clusters B-G overlap interaction classifications."
        ),

        "topic_modeling": wrapped(
            topic, statuses["topic_modeling"],
            "Topics, subtopics, assignments, distribution, timeline, and drift from topic-modeling artifact."
        ),

        "topic_interaction_analysis": {
            "status": "pending",
            "note": "Requires topic-segment timestamps aligned to overlap/interruption events. No causal claim is made by this script.",
        },

        "host_analysis": {
            "host_identification": wrapped(host_identification, statuses["host_identification"],
                "Host identity, top candidates, and evidence."),
            "host_actions": wrapped(host_actions,
                statuses["host_actions"] if loaded.get("host_actions") is not None else statuses["host_action"],
                "Host action classification; inspect completion/failure and label distributions before interpreting."),
            "host_participation": {
                "status": "pending",
                "note": "Host participation can be computed once the host identity is verified against speaker IDs."
            },
            "moderation_and_floor_management": wrapped(
                loaded.get("floor_management"), statuses["floor_management"],
                "Moderator/floor-management analysis."),
            "interruption_management": wrapped(
                loaded.get("interruption_management"), statuses["interruption_management"],
                "Host handling of interruptions, if produced."),
        },

        "speaker_interaction_analysis": {
            "status": "pending",
            "note": "Requires validated event-level speaker roles; transition counts alone do not establish interaction function.",
        },

        "validation_and_data_quality": {
            "status": "available",
            "source_file_status": statuses,
            "missing_source_files": [SOURCE_FILES[k] for k, s in statuses.items() if s == "not_available"],
            "read_errors": {k: v.get("_read_error") for k, v in loaded.items()
                            if isinstance(v, dict) and "_read_error" in v},
            "pipeline_turns_parsed": len(extract_turns(pipeline)) if pipeline else 0,
            "warning": "A source file being present does not imply its classification completed successfully; inspect its own status fields.",
        },

        "methodology_and_configuration": {
            "status": "available",
            "aggregation_script": "analysis/final_result.py",
            "transition_definition": "Adjacent turns from different speakers in chronological order.",
            "short_turn_threshold_seconds": 1.0,
            "overlap_duration_bins_seconds": {
                "very_short": [0.0, 0.2],
                "short": [0.2, 0.5],
                "moderate": [0.5, 1.0],
                "prolonged": "greater_than_1.0",
            },
            "overlap_merge_gap_seconds": get_path(cluster_data, "configuration", "merge_gap_seconds"),
            "prolonged_overlap_threshold_seconds": get_path(cluster_data, "configuration", "prolonged_overlap_threshold_seconds"),
            "principle": "Computed values are distinguished from unavailable, pending, or source-preserved analyses. No causal interpretation is inferred.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build final_result.json for MTP video analyses.")
    parser.add_argument("--videos-dir", default="data/v2/videos")
    parser.add_argument("--video-id", help="Process only one video folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final_result.json.")
    args = parser.parse_args()

    root = Path(args.videos_dir)
    if not root.exists():
        raise SystemExit(f"[ERROR] Videos directory not found: {root}")

    video_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.video_id:
        video_dirs = [p for p in video_dirs if p.name == args.video_id]
        if not video_dirs:
            raise SystemExit(f"[ERROR] No directory found for video ID: {args.video_id}")

    done = skipped = 0
    for video_dir in video_dirs:
        output_path = video_dir / OUTPUT_NAME
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {video_dir.name}: output exists (use --overwrite)")
            skipped += 1
            continue
        result = build_final_result(video_dir)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[OK] {video_dir.name} -> {output_path}")
        done += 1

    print(f"Finished. Processed={done}, skipped={skipped}, scanned={len(video_dirs)}")


if __name__ == "__main__":
    main()