"""Independent validation of ``discourse_summary.json`` (Phase 4).

The summary artifact reports discourse metrics; this module re-derives them from
the raw ``final.json`` with its own implementations and reports, per metric:

* an explicit definition, unit, denominator, provenance, and limitation;
* the value the summary reported and the independently re-derived value;
* whether the two agree, and the delta when they do not.

It also validates the raw event streams for the structural problems the audit
called out: out-of-order records, zero-duration records, exact duplicates, and
nested overlap candidates (audit L6).

The re-derivation intentionally duplicates the arithmetic in ``discourse_summary.py``
instead of importing it, so a shared bug cannot silently make both agree. The
only shared contract is the raw ``final.json`` schema and the configured
thresholds, which are read back from the summary artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "discourse-summary-review-1.0"
VERSION = "0.1.0"

DEFAULT_THRESHOLDS: dict[str, float] = {
    "merge_gap_s": 1.0,
    "short_turn_s": 1.0,
    "clean_transfer_gap_s": 0.1,
    "brief_backchannel_s": 1.0,
    "prolonged_overlap_s": 2.0,
    "floor_hold_s": 1.0,
    "floor_outcome_window_s": 2.0,
}

ACKS = {"yes", "yeah", "right", "exactly", "okay", "ok", "haan", "han",
        "हाँ", "हां", "जी", "ठीक", "बिल्कुल", "सही", "अच्छा"}

UNAVAILABLE = object()

NESTING_EXAMPLES_MAX = 5


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rnd(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def pct(numerator: float, denominator: Optional[float]) -> Optional[float]:
    return rnd(100 * numerator / denominator) if denominator and denominator > 0 else None


def per_minute(count: int, duration_s: Optional[float]) -> Optional[float]:
    return rnd(count * 60 / duration_s) if duration_s and duration_s > 0 else None


def issue(code: str, message: str, count: int = 1) -> dict[str, Any]:
    return {"code": code, "message": message, "count": count}


def value_at(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = document
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return UNAVAILABLE
    return node


def _timed(records: Any, start_key: str = "start", end_key: str = "end"):
    valid: list[tuple[int, float, float, dict[str, Any]]] = []
    zero_duration = 0
    if not isinstance(records, list):
        return valid, zero_duration
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        start, end = numeric(row.get(start_key)), numeric(row.get(end_key))
        if start is None or end is None:
            continue
        if end <= start:
            zero_duration += 1
            continue
        valid.append((index, start, end, row))
    return valid, zero_duration


def is_out_of_order(items: list[tuple[int, float, float, dict[str, Any]]]) -> bool:
    return any(items[i][1] < items[i - 1][1] for i in range(1, len(items)))


def validate_turns(raw: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid, zero_duration = _timed(raw)
    turns, rejected = [], 0
    for _, start, end, row in valid:
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            rejected += 1
            continue
        turns.append({"speaker": speaker, "start": start, "end": end})
    diagnostics = {"total_records": len(raw) if isinstance(raw, list) else 0, "valid": len(turns),
                   "rejected": rejected, "zero_duration": zero_duration, "out_of_order": is_out_of_order(valid)}
    return sorted(turns, key=lambda turn: (turn["start"], turn["end"], turn["speaker"])), diagnostics


def validate_overlaps(raw: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid, zero_duration = _timed(raw)
    events, rejected = [], 0
    for _, start, end, row in valid:
        speakers = row.get("speakers")
        if not isinstance(speakers, list):
            rejected += 1
            continue
        events.append({"start": start, "end": end, "duration": end - start,
                       "speakers": sorted({item for item in speakers if isinstance(item, str) and item}),
                       "case": row.get("case"), "source": row.get("source")})
    keys = [(event["start"], event["end"], tuple(event["speakers"]), event["case"], event["source"]) for event in events]
    duplicates = len(keys) - len(set(keys))
    sorted_events = sorted(events, key=lambda event: (event["start"], event["end"], str(event["case"])))
    diagnostics = {"total_records": len(raw) if isinstance(raw, list) else 0, "valid": len(events),
                   "rejected": rejected, "zero_duration": zero_duration, "out_of_order": is_out_of_order(valid),
                   "duplicates": duplicates, "nested": count_nested(sorted_events),
                   "nested_examples": nested_examples(sorted_events)}
    return sorted_events, diagnostics


def validate_interruptions(raw: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid, zero_duration = _timed(raw)
    events, rejected = [], 0
    for _, start, end, row in valid:
        interrupter, speakers = row.get("interrupter"), row.get("speakers")
        victims = sorted({item for item in speakers if isinstance(item, str) and item != interrupter}) if isinstance(speakers, list) else []
        if not isinstance(interrupter, str) or not victims:
            rejected += 1
            continue
        events.append({"start": start, "end": end, "duration": end - start, "interrupter": interrupter, "victims": victims})
    keys = [(event["start"], event["end"], event["interrupter"]) for event in events]
    duplicates = len(keys) - len(set(keys))
    diagnostics = {"total_records": len(raw) if isinstance(raw, list) else 0, "valid": len(events),
                   "rejected": rejected, "zero_duration": zero_duration, "out_of_order": is_out_of_order(valid),
                   "duplicates": duplicates}
    return sorted(events, key=lambda event: (event["start"], event["end"], event["interrupter"])), diagnostics


def count_nested(events: list[dict[str, Any]]) -> int:
    nested = 0
    for index, inner in enumerate(events):
        for other_index, outer in enumerate(events):
            if index == other_index:
                continue
            if outer["start"] <= inner["start"] and inner["end"] <= outer["end"]:
                nested += 1
                break
    return nested


def nested_examples(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for index, inner in enumerate(events):
        for other_index, outer in enumerate(events):
            if index == other_index:
                continue
            if outer["start"] <= inner["start"] and inner["end"] <= outer["end"]:
                examples.append({"inner": {"start": rnd(inner["start"]), "end": rnd(inner["end"]), "speakers": inner["speakers"]},
                                 "outer": {"start": rnd(outer["start"]), "end": rnd(outer["end"]), "speakers": outer["speakers"]}})
                break
        if len(examples) >= NESTING_EXAMPLES_MAX:
            break
    return examples


def merge_turns(turns: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and turn["speaker"] == merged[-1]["speaker"] and turn["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = max(merged[-1]["end"], turn["end"])
        else:
            merged.append(dict(turn))
    return merged


def transitions(turns: list[dict[str, Any]]) -> int:
    total, previous = 0, None
    for turn in turns:
        if previous is not None and turn["speaker"] != previous:
            total += 1
        previous = turn["speaker"]
    return total


def union_seconds(events: list[dict[str, Any]]) -> float:
    total, start, end = 0.0, None, None
    for event in sorted(events, key=lambda item: (item["start"], item["end"])):
        if start is None:
            start, end = event["start"], event["end"]
        elif event["start"] <= end:
            end = max(end, event["end"])
        else:
            total += end - start
            start, end = event["start"], event["end"]
    return total + (end - start if start is not None and end is not None else 0.0)


def episode_duration(data: dict[str, Any], turns: list[dict[str, Any]]) -> Optional[float]:
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    osd = data.get("osd_stats") if isinstance(data.get("osd_stats"), dict) else {}
    for value in (episode.get("duration_s"), osd.get("total_duration_s")):
        value = numeric(value)
        if value and value > 0:
            return value
    return max((turn["end"] for turn in turns), default=None) or None


def floor_outcome(event: dict[str, Any], turns: list[dict[str, Any]], t: dict[str, float]) -> str:
    deadline = event["end"] + t["floor_outcome_window_s"]
    holders = {turn["speaker"] for turn in turns if turn["start"] <= deadline and turn["end"] >= event["end"] + t["floor_hold_s"]}
    if len(holders) != 1:
        return "ambiguous"
    holder = next(iter(holders))
    return "successful" if holder == event["interrupter"] else "failed" if holder in event["victims"] else "ambiguous"


def is_acknowledgement(data: dict[str, Any], speaker: str, start: float, end: float) -> bool:
    words = []
    for segment in data.get("segments", []) if isinstance(data.get("segments"), list) else []:
        if not isinstance(segment, dict) or not isinstance(segment.get("words"), list):
            continue
        for word in segment["words"]:
            if not isinstance(word, dict) or word.get("speaker") != speaker:
                continue
            left, right = numeric(word.get("start")), numeric(word.get("end"))
            if left is not None and right is not None and right > start and left < end and isinstance(word.get("word"), str):
                words.append(word["word"].strip(" .,!?।").lower())
    return bool(words) and all(word in ACKS for word in words)


def competitive(event: dict[str, Any], turns: list[dict[str, Any]], t: dict[str, float]) -> bool:
    if event["duration"] < t["brief_backchannel_s"]:
        return False
    continuing = {turn["speaker"] for turn in turns if turn["start"] < event["end"] and turn["end"] >= event["end"] + t["floor_hold_s"]}
    return len(continuing.intersection(event["speakers"])) >= 2


def rederive(data: dict[str, Any], t: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    turns, turn_diag = validate_turns(data.get("speaker_turns"))
    overlaps, overlap_diag = validate_overlaps(data.get("overlap_candidates"))
    interruptions, interruption_diag = validate_interruptions(data.get("interruptions"))
    duration_s = episode_duration(data, turns)
    merged = merge_turns(turns, t["merge_gap_s"])
    outcomes = Counter(floor_outcome(event, turns, t) for event in interruptions)
    metrics = {
        "episode_duration_s": rnd(duration_s),
        "number_of_speakers": len({turn["speaker"] for turn in merged}),
        "total_valid_turns": len(turns),
        "total_merged_turn_s": rnd(sum(turn["end"] - turn["start"] for turn in merged)),
        "total_speaker_transitions": transitions(merged),
        "transitions_per_minute": per_minute(transitions(merged), duration_s),
        "overlap_candidate_count": len(overlaps),
        "unique_elapsed_overlap_s": rnd(union_seconds(overlaps)),
        "cumulative_candidate_overlap_s": rnd(sum(event["duration"] for event in overlaps)),
        "overlap_ratio_pct": pct(union_seconds(overlaps), duration_s),
        "interruption_attempt_count": len(interruptions),
        "successful_interruption_count": outcomes["successful"],
        "failed_interruption_count": outcomes["failed"],
        "ambiguous_interruption_count": outcomes["ambiguous"],
        "clean_turn_transfer_count": sum(
            left["speaker"] != right["speaker"] and 0 <= right["start"] - left["end"] <= t["clean_transfer_gap_s"]
            for left, right in zip(turns, turns[1:])),
        "brief_backchannel_count": sum(
            any(is_acknowledgement(data, speaker, event["start"], event["end"]) for speaker in event["speakers"])
            for event in overlaps),
        "competitive_overlap_count": sum(competitive(event, turns, t) for event in overlaps),
        "prolonged_overlap_count": sum(event["duration"] >= t["prolonged_overlap_s"] for event in overlaps),
    }
    validation = {"speaker_turns": turn_diag, "overlap_candidates": overlap_diag, "interruptions": interruption_diag}
    return metrics, validation


METRIC_SPECS: list[dict[str, Any]] = [
    {"metric": "episode_duration_s", "definition": "Episode length used as every rate denominator.",
     "unit": "seconds", "denominator": "n/a", "provenance": "episode.duration_s, else osd_stats.total_duration_s, else last valid turn end",
     "limitation": "The last-turn-end fallback is a proxy, not true media duration.",
     "reported_path": ("metadata", "duration_s"), "rederived_key": "episode_duration_s"},
    {"metric": "number_of_speakers", "definition": "Distinct diarization labels after same-speaker turn merge.",
     "unit": "count", "denominator": "n/a", "provenance": "speaker_turns[].speaker",
     "limitation": "Diarization labels need not correspond to people.",
     "reported_path": ("participation", "number_of_speakers"), "rederived_key": "number_of_speakers"},
    {"metric": "total_valid_turns", "definition": "Chronologically valid speaker turns kept for analysis.",
     "unit": "count", "denominator": "raw speaker_turns records", "provenance": "speaker_turns[]",
     "limitation": "Invalid records are silently excluded from every downstream metric.",
     "reported_path": ("quality", "input_counts", "valid_turns"), "rederived_key": "total_valid_turns"},
    {"metric": "total_merged_turn_s", "definition": "Summed diarized speaking time over merged turns.",
     "unit": "seconds", "denominator": "n/a", "provenance": "speaker_turns[] after merge",
     "limitation": "Overlapping turns can exceed wall-clock speech time.",
     "reported_path": ("participation", "total_merged_turn_s"), "rederived_key": "total_merged_turn_s"},
    {"metric": "total_speaker_transitions", "definition": "Adjacent speaker changes in the merged turn sequence.",
     "unit": "count", "denominator": "merged turns", "provenance": "merged speaker_turns[]",
     "limitation": "Merge gap is a heuristic; it changes the transition count.",
     "reported_path": ("switching", "total_speaker_transitions"), "rederived_key": "total_speaker_transitions"},
    {"metric": "transitions_per_minute", "definition": "Speaker transitions normalized by episode duration.",
     "unit": "events per minute", "denominator": "episode_duration_s", "provenance": "merged speaker_turns[] / duration",
     "limitation": "Inherits duration-source and merge-gap uncertainty.",
     "reported_path": ("switching", "transitions_per_minute"), "rederived_key": "transitions_per_minute"},
    {"metric": "overlap_candidate_count", "definition": "Detector-reconciliation overlap candidates kept.",
     "unit": "count", "denominator": "raw overlap_candidates records", "provenance": "overlap_candidates[]",
     "limitation": "Candidates are not confirmed or human-validated overlap events.",
     "reported_path": ("overlap", "candidate_event_count"), "rederived_key": "overlap_candidate_count"},
    {"metric": "unique_elapsed_overlap_s", "definition": "Elapsed overlap time after interval union.",
     "unit": "seconds", "denominator": "episode_duration_s", "provenance": "temporal union of overlap_candidates[]",
     "limitation": "Nested/duplicate candidates collapse; source counts are preserved separately.",
     "reported_path": ("overlap", "unique_elapsed_overlap_s"), "rederived_key": "unique_elapsed_overlap_s"},
    {"metric": "cumulative_candidate_overlap_s", "definition": "Sum of candidate durations before union.",
     "unit": "seconds", "denominator": "n/a", "provenance": "sum of overlap_candidates[].duration",
     "limitation": "Double counts nested and duplicate candidates.",
     "reported_path": ("overlap", "cumulative_candidate_overlap_s"), "rederived_key": "cumulative_candidate_overlap_s"},
    {"metric": "overlap_ratio_pct", "definition": "Unique elapsed overlap time as a share of episode duration.",
     "unit": "percent (0-100)", "denominator": "episode_duration_s", "provenance": "union(overlap_candidates[]) / duration",
     "limitation": "Candidate-based; not a validated overlap prevalence.",
     "reported_path": ("overlap", "overlap_ratio_pct"), "rederived_key": "overlap_ratio_pct"},
    {"metric": "interruption_attempt_count", "definition": "Valid OSD-derived interruption attempts.",
     "unit": "count", "denominator": "raw interruptions records", "provenance": "interruptions[]",
     "limitation": "OSD-derived operational attempts, not human-validated interruptions.",
     "reported_path": ("interruptions", "total_attempts"), "rederived_key": "interruption_attempt_count"},
    {"metric": "successful_interruption_count", "definition": "Attempts where exactly the interrupter holds the post-overlap floor.",
     "unit": "count", "denominator": "successful + failed attempts", "provenance": "rule_based floor outcome",
     "limitation": "Rule-based, threshold-sensitive; ambiguous attempts are excluded from the rate.",
     "reported_path": ("event_classification", "successful_interruption", "count"), "rederived_key": "successful_interruption_count"},
    {"metric": "failed_interruption_count", "definition": "Attempts where exactly a victim holds the post-overlap floor.",
     "unit": "count", "denominator": "successful + failed attempts", "provenance": "rule_based floor outcome",
     "limitation": "Rule-based, threshold-sensitive; ambiguous attempts are excluded from the rate.",
     "reported_path": ("event_classification", "failed_interruption", "count"), "rederived_key": "failed_interruption_count"},
    {"metric": "ambiguous_interruption_count", "definition": "Attempts with no unique post-overlap floor holder.",
     "unit": "count", "denominator": "all valid attempts", "provenance": "rule_based floor outcome",
     "limitation": "Large ambiguous share means the success/failure rule is inconclusive.",
     "reported_path": ("event_classification", "unclassified_or_ambiguous_interruption_count"), "rederived_key": "ambiguous_interruption_count"},
    {"metric": "clean_turn_transfer_count", "definition": "Different-speaker transitions within the non-overlap gap.",
     "unit": "count", "denominator": "total_speaker_transitions", "provenance": "rule_based, raw turns",
     "limitation": "Threshold-dependent; measures smoothness, not politeness.",
     "reported_path": ("event_classification", "clean_turn_transfer", "count"), "rederived_key": "clean_turn_transfer_count"},
    {"metric": "brief_backchannel_count", "definition": "Overlap candidates whose words are all acknowledgement tokens.",
     "unit": "count", "denominator": "overlap_candidate_count", "provenance": "rule_based lexicon",
     "limitation": "Lexicon is English/Hindi and small; ASR errors and missing word timings hide backchannels.",
     "reported_path": ("event_classification", "brief_backchannel", "count"), "rederived_key": "brief_backchannel_count"},
    {"metric": "competitive_overlap_count", "definition": "Overlaps where two involved speakers both continue past overlap end.",
     "unit": "count", "denominator": "overlap_candidate_count", "provenance": "rule_based, raw turns",
     "limitation": "Depends on floor_hold_s and diarization quality.",
     "reported_path": ("event_classification", "competitive_overlap", "count"), "rederived_key": "competitive_overlap_count"},
    {"metric": "prolonged_overlap_count", "definition": "Overlap candidates at or above the prolonged-overlap threshold.",
     "unit": "count", "denominator": "overlap_candidate_count", "provenance": "rule_based, candidate durations",
     "limitation": "Duration alone is not a semantic judgment.",
     "reported_path": ("event_classification", "prolonged_overlap", "count"), "rederived_key": "prolonged_overlap_count"},
    {"metric": "agreement_overlap_rate_pct", "definition": "Share of overlaps expressing semantic agreement.",
     "unit": "percent (0-100)", "denominator": "overlap_candidate_count", "provenance": "requires semantic annotation/classifier",
     "limitation": "Not computable: candidate.case=AGREE is detector corroboration, never semantic agreement.",
     "reported_path": ("event_classification", "agreement_overlap", "rate_pct"), "rederived_key": None},
    {"metric": "disagreement_overlap_rate_pct", "definition": "Share of overlaps expressing semantic disagreement.",
     "unit": "percent (0-100)", "denominator": "overlap_candidate_count", "provenance": "requires semantic annotation/classifier",
     "limitation": "Not computable without annotations or a validated classifier.",
     "reported_path": ("event_classification", "disagreement_overlap", "rate_pct"), "rederived_key": None},
]


def metric_rows(reported: dict[str, Any], rederived: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for spec in METRIC_SPECS:
        reported_value = value_at(reported, spec["reported_path"]) if spec.get("reported_path") else UNAVAILABLE
        if reported_value is UNAVAILABLE:
            reported_value = None
        if spec.get("rederived_key") is None:
            availability, rederived_value, match, delta = "unavailable_by_design", None, None, None
        else:
            rederived_value = rederived.get(spec["rederived_key"])
            if isinstance(reported_value, (int, float)) and isinstance(rederived_value, (int, float)):
                delta = rnd(abs(float(reported_value) - float(rederived_value)))
                match, availability = delta == 0.0, "available"
            elif reported_value is None:
                match, availability, delta = None, "not_reported", None
            else:
                match, availability, delta = None, "not_rederivable", None
        rows.append({"metric": spec["metric"], "definition": spec["definition"], "unit": spec["unit"],
                     "denominator": spec["denominator"], "provenance": spec["provenance"], "limitation": spec["limitation"],
                     "reported_value": reported_value, "rederived_value": rederived_value,
                     "match": match, "delta": delta, "availability": availability})
    return rows


def load_final(directory: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    path = directory / "final.json"
    if not path.is_file():
        return None, "missing_final_json"
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_final_json"
    return (data, None) if isinstance(data, dict) else (None, "invalid_top_level_schema")


def review_episode(directory_id: str, data: dict[str, Any], reported: Optional[dict[str, Any]],
                   thresholds: dict[str, float]) -> dict[str, Any]:
    metrics, validation = rederive(data, thresholds)
    rows = metric_rows(reported or {}, metrics)
    available = [row for row in rows if row["availability"] == "available"]
    mismatches = [row["metric"] for row in available if row["match"] is False]
    warnings = []
    structural = {"out_of_order": sum(validation[name].get("out_of_order", 0) for name in validation),
                  "zero_duration": sum(validation[name].get("zero_duration", 0) for name in validation),
                  "duplicates": sum(validation[name].get("duplicates", 0) for name in validation)}
    if structural["out_of_order"]:
        warnings.append(issue("out_of_order_events", "Raw events are not chronologically ordered", structural["out_of_order"]))
    if structural["zero_duration"]:
        warnings.append(issue("zero_duration_events", "Raw events have non-positive duration", structural["zero_duration"]))
    if validation["overlap_candidates"]["duplicates"] or validation["interruptions"]["duplicates"]:
        warnings.append(issue("duplicate_events", "Exact duplicate events present; union handles overlap double counting",
                              validation["overlap_candidates"]["duplicates"] + validation["interruptions"]["duplicates"]))
    if validation["overlap_candidates"]["nested"]:
        warnings.append(issue("nested_overlap_candidates", "Overlap candidates nested inside another candidate (audit L6)",
                              validation["overlap_candidates"]["nested"]))
    if mismatches:
        warnings.append(issue("metric_mismatch", "Re-derived value differs from the reported value", len(mismatches)))
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    return {
        "metadata": {"video_id": episode.get("video_id", directory_id), "directory_id": directory_id,
                     "title": episode.get("title"), "reported_in_summary": reported is not None},
        "status": "reviewed" if reported is not None else "not_in_summary",
        "metrics": rows,
        "validation": validation,
        "structural_summary": structural,
        "cross_check": {"metrics_compared": len(available), "matched": len(available) - len(mismatches),
                        "mismatched": len(mismatches), "mismatched_metrics": mismatches},
        "warnings": warnings,
    }


def build_review(videos_dir: Path, summary: dict[str, Any], thresholds: Optional[dict[str, float]] = None) -> dict[str, Any]:
    metadata = summary.get("report_metadata") if isinstance(summary.get("report_metadata"), dict) else {}
    thresholds = thresholds or (metadata.get("thresholds") if isinstance(metadata.get("thresholds"), dict) else None) or DEFAULT_THRESHOLDS
    reported_episodes = {str(item.get("metadata", {}).get("directory_id")): item for item in summary.get("episodes", [])
                         if isinstance(item, dict)} if isinstance(summary.get("episodes"), list) else {}
    episodes = []
    not_reviewed = []
    for directory in sorted(path for path in videos_dir.iterdir() if path.is_dir()) if videos_dir.is_dir() else []:
        data, reason = load_final(directory)
        if data is None:
            not_reviewed.append({"directory_id": directory.name, "reason": reason})
            continue
        episodes.append(review_episode(directory.name, data, reported_episodes.get(directory.name), thresholds))
    compared = sum(item["cross_check"]["metrics_compared"] for item in episodes)
    matched = sum(item["cross_check"]["matched"] for item in episodes)
    mismatched = sum(item["cross_check"]["mismatched"] for item in episodes)
    warning_counts = Counter(item["code"] for episode in episodes for item in episode["warnings"])
    return {
        "schema_version": SCHEMA_VERSION,
        "report_metadata": {
            "analysis_version": VERSION,
            "method": "Re-derive each discourse metric from raw final.json independently, then compare to discourse_summary.json.",
            "independence_note": "This module re-implements the arithmetic; it does not import discourse_summary.py.",
            "input_directory": str(videos_dir),
            "summary_schema_version": metadata.get("schema_version"),
            "thresholds": thresholds,
            "units": {"timestamps_and_durations": "seconds", "rates": "events per minute", "percentages": "0-100"},
            "semantic_label_policy": "Never infer agreement/disagreement from overlap or candidate.case.",
        },
        "metric_definitions": [{key: spec[key] for key in ("metric", "definition", "unit", "denominator", "provenance", "limitation")}
                               for spec in METRIC_SPECS],
        "episodes": episodes,
        "aggregate": {"processing": {"reviewed": len(episodes), "not_reviewed": len(not_reviewed)},
                      "cross_check": {"metrics_compared": compared, "matched": matched, "mismatched": mismatched,
                                      "match_rate_pct": pct(matched, compared)},
                      "quality_warning_counts": dict(sorted(warning_counts.items()))},
        "not_reviewed": not_reviewed,
        "limitations": [
            "Re-derivation shares the raw final.json schema and configured thresholds with the summary.",
            "Nested/duplicate overlap candidates collapse in the union but persist in source counts (audit L6).",
            "Agreement/disagreement remain unavailable without annotations or a validated classifier.",
            "A matched metric confirms arithmetic agreement, not semantic correctness of the underlying detector.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate discourse_summary.json against raw final.json.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--summary", type=Path, default=Path("data/v2/analysis/discourse_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/analysis/discourse_summary_review.json"))
    args = parser.parse_args()

    if not args.summary.is_file():
        parser.error(f"summary artifact not found: {args.summary} (run the discourse summary first)")
    with args.summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    review = build_review(args.videos_dir, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(review, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    cross = review["aggregate"]["cross_check"]
    print(f"Discourse review: {cross['matched']}/{cross['metrics_compared']} metrics matched "
          f"({cross['mismatched']} mismatched) across {review['aggregate']['processing']['reviewed']} episode(s)")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
