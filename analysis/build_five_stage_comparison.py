#!/usr/bin/env python3
"""Build ``comparison/five_stage_comparison.json`` from existing V2 artifacts.

This is a read-only aggregator: it never changes final.json, merge_final.json,
or any analysis artifact.  It deliberately keeps attributed speech duration
separate from floor-holding duration, because a merged interval can contain a
pause.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

STAGES = (
    ("original", "S1: original speaker turns before merging", Path(".")),
    ("merged_no_threshold", "S2: baseline merged, no threshold", Path("baseline")),
    ("threshold_0.5", "S3: 0.5 second threshold and fresh merge", Path("threshold_0.5")),
    ("threshold_1.0", "S4: 1.0 second threshold and fresh merge", Path("threshold_1.0")),
    ("threshold_2.0", "S5: 2.0 second threshold and fresh merge", Path("threshold_2.0")),
)


def load(path: Path, missing: list, label: str) -> Optional[dict]:
    if not path.is_file():
        missing.append({"source": label, "path": str(path), "reason": "not found"})
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        missing.append({"source": label, "path": str(path), "reason": str(exc)})
        return None
    if not isinstance(value, dict):
        missing.append({"source": label, "path": str(path), "reason": "expected JSON object"})
        return None
    return value


def num(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def interval(record: dict) -> Optional[tuple[float, float]]:
    start, end = num(record.get("start")), num(record.get("end"))
    return (start, end) if start is not None and end is not None and end > start else None


def summary(values: Iterable[float]) -> dict:
    values = [round(v, 3) for v in values if v is not None and v >= 0]
    if not values:
        return {"count": 0, "total": 0.0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "total": round(sum(values), 3), "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3), "min": round(min(values), 3), "max": round(max(values), 3)}


def turns(data: Optional[dict]) -> list[dict]:
    result = []
    for item in (data or {}).get("speaker_turns", []):
        if not isinstance(item, dict) or not item.get("speaker"):
            continue
        pair = interval(item)
        if pair:
            result.append({"speaker": str(item["speaker"]), "start": pair[0], "end": pair[1]})
    return sorted(result, key=lambda x: (x["start"], x["end"]))


def overlap_records(data: Optional[dict]) -> list[dict]:
    result = []
    for item in (data or {}).get("overlap_candidates", []):
        if not isinstance(item, dict):
            continue
        pair = interval(item)
        if pair:
            speakers = item.get("speakers", [])
            result.append({"start": pair[0], "end": pair[1], "duration": pair[1] - pair[0],
                           "speakers": sorted(str(x) for x in speakers) if isinstance(speakers, list) else []})
    return result


def speaker_stats(source_turns: list[dict], continuity: list[dict], transition_totals: dict) -> dict:
    by_speaker = defaultdict(list)
    for turn in source_turns:
        by_speaker[turn["speaker"]].append(turn["end"] - turn["start"])
    continuous = defaultdict(list)
    for turn in continuity:
        speaker = turn.get("speaker")
        duration = num(turn.get("floor_holding_duration", turn.get("end", 0) - turn.get("start", 0)))
        speech = num(turn.get("speech_duration"))
        if speaker and duration is not None:
            continuous[str(speaker)].append((duration, speech))
    total = sum(sum(values) for values in by_speaker.values())
    all_speakers = sorted(set(by_speaker) | set(continuous) | set(transition_totals))
    output = {}
    for speaker in all_speakers:
        attributed = sum(by_speaker[speaker])
        episodes = continuous[speaker]
        held = [x[0] for x in episodes]
        output[speaker] = {
            "attributed_speaking_seconds": round(attributed, 3),
            "speech_share_percent_of_attributed_total": round(attributed / total * 100, 3) if total else None,
            "original_turn_frequency": len(by_speaker[speaker]),
            "continuous_speaking_episode_count": len(episodes),
            "continuous_floor_holding_duration": summary(held),
            "speaker_switches": transition_totals.get(speaker),
        }
    return {"share_denominator": "sum of attributed source-turn durations; overlap is counted for each involved speaker",
            "total_attributed_speaking_seconds": round(total, 3), "by_speaker": output}


def transition_stats(analysis: Optional[dict], duration: Optional[float]) -> dict:
    body = (analysis or {}).get("transition_analysis", {})
    matrix = body.get("confirmed_transition_counts") if isinstance(body, dict) else None
    matrix = matrix if isinstance(matrix, dict) else {}
    total = sum(v for v in matrix.values() if isinstance(v, int))
    pairs = [{"transition": key, "count": value} for key, value in sorted(matrix.items())]
    nonzero = [x for x in pairs if x["count"] > 0]
    speaker_totals = defaultdict(lambda: {"incoming": 0, "outgoing": 0})
    for row in pairs:
        if " -> " in row["transition"]:
            left, right = row["transition"].split(" -> ", 1)
            speaker_totals[left]["outgoing"] += row["count"]
            speaker_totals[right]["incoming"] += row["count"]
    return {"definition": "confirmed directed transitions from speaker_transition_analysis; self transitions excluded",
            "total_speaker_transitions": total, "total_speaker_switches": total,
            "transitions_per_minute": round(total / (duration / 60), 3) if duration else None,
            "pairwise_frequency": pairs, "unique_pairs": len(nonzero),
            "most_frequent": max(nonzero, key=lambda x: x["count"]) if nonzero else None,
            "least_frequent_nonzero": min(nonzero, key=lambda x: x["count"]) if nonzero else None,
            "by_speaker": dict(speaker_totals)}


def overlap_stats(records: list[dict], duration: Optional[float], cluster: Optional[dict]) -> dict:
    durations = [x["duration"] for x in records]
    categories = Counter("very_short" if d < .2 else "short" if d < .5 else "moderate" if d <= 1 else "prolonged" for d in durations)
    pairs = defaultdict(list)
    for row in records:
        if len(row["speakers"]) >= 2:
            pairs[" | ".join(row["speakers"])].append(row["duration"])
    cluster_regions = (cluster or {}).get("merged_overlap_regions")
    debate = ((cluster or {}).get("statistics", {}).get("overlap_interaction_clusters", {}).get("debate_level"))
    return {"total_overlap_events": len(records), "duration": summary(durations),
            "overlap_ratio_percent_of_episode": round(sum(durations) / duration * 100, 3) if duration else None,
            "duration_categories": {key: categories.get(key, 0) for key in ("very_short", "short", "moderate", "prolonged")},
            "continuous_overlap_events": len(cluster_regions) if isinstance(cluster_regions, list) else None,
            "speaker_pairs": [{"pair": pair, **summary(values)} for pair, values in sorted(pairs.items())],
            "cluster_classification": debate if isinstance(debate, dict) else None}


def gap_stats(gap: Optional[dict]) -> dict:
    if not gap:
        return {"available": False, "reason": "turn_gap_analysis.json unavailable"}
    rows = gap.get("inter_turn_gaps", [])
    rows = [x for x in rows if isinstance(x, dict)]
    positive = [num(x.get("duration")) for x in rows if num(x.get("duration")) and num(x.get("duration")) > 0]
    types = Counter(str(x.get("gap_type")) for x in rows)
    debate = gap.get("statistics", {}).get("debate_level", {})
    return {"available": True, "positive_silence_gap_duration": summary(positive), "gap_type_counts": dict(types),
            "gaps_eliminated_by_gap_merge": debate.get("turns_merged"),
            "overlapping_turn_pairs": debate.get("overlapping_turn_pairs"), "gaps_with_overlap_events": debate.get("gaps_with_overlap_events"),
            "note": "Positive gaps are pairwise inter-turn gaps from the existing gap analysis; they are not union-of-silence coverage."}


def cluster_stats(cluster: Optional[dict]) -> dict:
    if not cluster:
        return {"available": False, "clusters": {key: None for key in "ABCDEFGH"}}
    debate = cluster.get("statistics", {}).get("overlap_interaction_clusters", {}).get("debate_level", {})
    counts = debate.get("cluster_counts", {})
    percentages = debate.get("cluster_percentages", {})
    transitions = cluster.get("statistics", {}).get("speaker_transitions", {})
    output = {"A": {"count": transitions.get("cluster_A_candidate_count"), "percentage": transitions.get("cluster_A_candidate_percentage")}}
    for key in "BCDEFG": output[key] = {"count": counts.get(key), "percentage": percentages.get(key)}
    output["H"] = {"count": None, "percentage": None, "reason": "not defined by current cluster artifact"}
    return {"available": True, "clusters": output, "speaker_level": cluster.get("statistics", {}).get("overlap_interaction_clusters", {}).get("speaker_level")}


def changes(value: Optional[float], reference: Optional[float]) -> dict:
    if value is None or reference is None: return {"absolute": None, "percent": None}
    delta = value - reference
    return {"absolute": round(delta, 3), "percent": round(delta / reference * 100, 3) if reference else None}


def build_stage(video_dir: Path, stage_id: str, description: str, rel: Path, missing: list) -> dict:
    directory = video_dir / rel
    final = load(directory / "final.json", missing, f"{stage_id}.final")
    merged = load(directory / "merge_final.json", missing, f"{stage_id}.merge_final") if stage_id != "original" else None
    gap = load(directory / "turn_gap_analysis.json", missing, f"{stage_id}.turn_gap")
    transition_path = directory / "speaker_transition_analysis.json"
    if stage_id == "original" and not transition_path.exists(): transition_path = directory / "analysis" / "speaker_transition_analysis.json"
    transition = load(transition_path, missing, f"{stage_id}.transitions")
    cluster = load(directory / "overlap_cluster_analysis.json", missing, f"{stage_id}.clusters")
    host = load(directory / "host_identification" / "host_identification.json", missing, f"{stage_id}.host")
    source_turns = turns(final)
    analysis_turns = turns(merged or final)
    continuity = (gap or {}).get("merged_turns", [])
    duration = num((final or {}).get("episode", {}).get("duration_s"))
    transitions = transition_stats(transition, duration)
    participation = speaker_stats(source_turns, continuity if isinstance(continuity, list) else [], transitions["by_speaker"])
    records = overlap_records(final)
    host_id = (host or {}).get("predicted_host_speaker")
    host_metrics = participation["by_speaker"].get(host_id) if host_id else None
    return {"description": description, "source_files": {"final": str(directory / "final.json"), "merged": str(directory / "merge_final.json") if merged else None,
            "gap": str(directory / "turn_gap_analysis.json"), "transitions": str(transition_path), "clusters": str(directory / "overlap_cluster_analysis.json")},
            "timeline_statistics": {"episode_duration_seconds": duration, "source_speaker_turn_count": len(source_turns), "analysis_turn_count": len(analysis_turns),
                "merged_speaker_turn_count": len(analysis_turns) if merged else None, "attributed_speech_seconds": participation["total_attributed_speaking_seconds"],
                "timeline_span_seconds": round(max((x["end"] for x in analysis_turns), default=0) - min((x["start"] for x in analysis_turns), default=0), 3) if analysis_turns else None},
            "speaker_participation": participation, "speaker_continuity": {"episode_level": summary([num(x.get("floor_holding_duration")) for x in continuity if isinstance(x, dict)]), "definition": "gap-analysis merged turn / floor-holding interval"},
            "overlap_statistics": overlap_stats(records, duration, cluster), "speaker_transitions": transitions, "gap_analysis": gap_stats(gap),
            "interaction_clusters": cluster_stats(cluster), "host_analysis": {"identified_host_speaker": host_id, "confidence_band": (host or {}).get("confidence_band"), "participation": host_metrics, "status": "heuristic, not human verified" if host else "unavailable"},
            "interruption_analysis": {"candidate_count": len((final or {}).get("interruptions", [])) if final else None, "status": "candidate events only; no stage-local interruption-management artifact"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--video-id")
    args = parser.parse_args()
    video_dirs = [args.data_dir / args.video_id] if args.video_id else sorted(p.parent for p in args.data_dir.glob("*/final.json"))
    for video_dir in video_dirs:
        missing = []
        advertisements = load(video_dir / "advertisement_analysis" / "advertisement_exclusions.json", missing, "advertisement_exclusions")
        stages = {sid: build_stage(video_dir, sid, desc, rel, missing) for sid, desc, rel in STAGES}
        original, baseline = stages["original"], stages["merged_no_threshold"]
        impact = {"merge_S1_to_S2": {"turns_removed": (original["timeline_statistics"]["source_speaker_turn_count"] - baseline["timeline_statistics"]["analysis_turn_count"]),
            "attributed_speech_seconds_change": changes(baseline["timeline_statistics"]["attributed_speech_seconds"], original["timeline_statistics"]["attributed_speech_seconds"]),
            "timeline_span_change": changes(baseline["timeline_statistics"]["timeline_span_seconds"], original["timeline_statistics"]["timeline_span_seconds"])} , "thresholds": {}, "cumulative_from_S1": {}}
        for key, label in (("threshold_0.5", "0.5"), ("threshold_1.0", "1.0"), ("threshold_2.0", "2.0")):
            current = stages[key]
            impact["thresholds"][label] = {"overlap_events_removed_from_S2": baseline["overlap_statistics"]["total_overlap_events"] - current["overlap_statistics"]["total_overlap_events"],
                "overlap_duration_change_from_S2": changes(current["overlap_statistics"]["duration"]["total"], baseline["overlap_statistics"]["duration"]["total"]),
                "additional_turns_removed_after_threshold": baseline["timeline_statistics"]["analysis_turn_count"] - current["timeline_statistics"]["analysis_turn_count"]}
            impact["cumulative_from_S1"][label] = {"turn_count_change": changes(current["timeline_statistics"]["analysis_turn_count"], original["timeline_statistics"]["source_speaker_turn_count"]), "overlap_duration_change": changes(current["overlap_statistics"]["duration"]["total"], original["overlap_statistics"]["duration"]["total"])}
        output = {"schema_version": "five-stage-comparison-1.0", "metadata": {"video_id": video_dir.name, "video_title": (stages["original"].get("source_files") and (load(video_dir / "final.json", [], "metadata") or {}).get("episode", {}).get("title")), "language": "Hindi", "advertisement_exclusion": (advertisements or {}).get("summary")},
            "comparison_metadata": {"stages": [x[0] for x in STAGES], "generated_at_utc": datetime.now(timezone.utc).isoformat(), "metric_definitions": {"speech_share": "attributed turn duration / total attributed turn duration", "switch": "same as confirmed transition", "overlap_duration": "sum of stage-local overlap candidate durations"}},
            "stage_results": stages, "processing_impact": impact, "data_quality": {"unavailable_sources": missing, "validation_warnings": ["Pairwise gap totals are not union-of-silence coverage.", "Host predictions and clusters are not human-verified."]}}
        out = video_dir / "comparison" / "five_stage_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[OK] {video_dir.name}: {out}")


if __name__ == "__main__": main()
