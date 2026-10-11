"""Create a reproducible discourse report from v2 final.json artifacts.

The input has diarized turns, detector-reconciliation overlap candidates, and
OSD-derived interruption attempts. It has no human semantic labels: AGREE is
detector agreement, never conversational agreement.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Thresholds:
    """Configurable heuristic thresholds in seconds."""
    merge_gap_s: float = 1.0
    short_turn_s: float = 1.0
    clean_transfer_gap_s: float = 0.1
    brief_backchannel_s: float = 1.0
    prolonged_overlap_s: float = 2.0
    floor_hold_s: float = 1.0
    floor_outcome_window_s: float = 2.0


DEFAULTS = Thresholds()
ACKS = {"yes", "yeah", "right", "exactly", "okay", "ok", "haan", "han",
        "हाँ", "हां", "जी", "ठीक", "बिल्कुल", "सही", "अच्छा"}


def numeric(value: Any) -> Optional[float]:
    """Return a finite number, never treating bool as a timestamp."""
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


def mean(values: list[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return rnd(statistics.fmean(usable)) if usable else None


def median(values: list[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return rnd(statistics.median(usable)) if usable else None


def load_episodes(root: Path) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load documents deterministically and preserve bad-file diagnostics."""
    loaded, skipped, failed = [], [], []
    if not root.is_dir():
        return loaded, skipped, [{"path": str(root), "reason": "input_directory_missing"}]
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "final.json"
        if not path.is_file():
            skipped.append({"video_id": directory.name, "path": str(path), "reason": "missing_final_json"})
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            failed.append({"video_id": directory.name, "path": str(path), "reason": "invalid_json", "detail": str(exc)})
            continue
        if isinstance(data, dict):
            loaded.append((directory.name, data))
        else:
            failed.append({"video_id": directory.name, "path": str(path), "reason": "invalid_top_level_schema"})
    return loaded, skipped, failed


def validate_turns(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings = []
    if not isinstance(raw, list):
        return [], [issue("missing_or_invalid_speaker_turns", "speaker_turns is not a list")]
    turns, bad = [], 0
    for row in raw:
        if not isinstance(row, dict):
            bad += 1
            continue
        speaker, start, end = row.get("speaker"), numeric(row.get("start")), numeric(row.get("end"))
        if not isinstance(speaker, str) or not speaker.strip() or start is None or end is None or end <= start:
            bad += 1
        else:
            turns.append({"speaker": speaker, "start": start, "end": end})
    if bad:
        warnings.append(issue("invalid_turn_excluded", "Missing speaker, non-finite timestamp, or nonpositive duration", bad))
    if any(turns[index]["start"] < turns[index - 1]["start"] for index in range(1, len(turns))):
        warnings.append(issue("turns_out_of_order", "Turns were sorted chronologically for analysis"))
    return sorted(turns, key=lambda turn: (turn["start"], turn["end"], turn["speaker"])), warnings


def merge_turns(turns: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    merged = []
    for turn in turns:
        if merged and turn["speaker"] == merged[-1]["speaker"] and turn["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = max(merged[-1]["end"], turn["end"])
        else:
            merged.append(dict(turn))
    return merged


def duration(data: dict[str, Any], turns: list[dict[str, Any]]) -> tuple[Optional[float], Optional[str]]:
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    osd = data.get("osd_stats") if isinstance(data.get("osd_stats"), dict) else {}
    for source, value in (("episode.duration_s", episode.get("duration_s")), ("osd_stats.total_duration_s", osd.get("total_duration_s"))):
        value = numeric(value)
        if value and value > 0:
            return value, source
    return (max(turn["end"] for turn in turns), "last_valid_turn_end_proxy") if turns else (None, None)


def participation(raw: list[dict[str, Any]], merged: list[dict[str, Any]]) -> dict[str, Any]:
    raw_s, raw_n, merged_s, merged_n = Counter(), Counter(), Counter(), Counter()
    for turns, seconds, counts in ((raw, raw_s, raw_n), (merged, merged_s, merged_n)):
        for turn in turns:
            seconds[turn["speaker"]] += turn["end"] - turn["start"]
            counts[turn["speaker"]] += 1
    total = sum(merged_s.values())
    by_speaker = {
        speaker: {
            "raw_turn_count": raw_n[speaker], "merged_turn_count": merged_n[speaker],
            "raw_speaking_s": rnd(raw_s[speaker]), "merged_speaking_s": rnd(merged_s[speaker]),
            "speech_share_pct": pct(merged_s[speaker], total),
            "mean_raw_turn_s": rnd(raw_s[speaker] / raw_n[speaker]) if raw_n[speaker] else None,
            "mean_merged_turn_s": rnd(merged_s[speaker] / merged_n[speaker]) if merged_n[speaker] else None,
        } for speaker in sorted(merged_s)
    }
    return {"number_of_speakers": len(by_speaker), "speakers": list(by_speaker),
            "total_raw_turn_s": rnd(sum(raw_s.values())), "total_merged_turn_s": rnd(total),
            "by_speaker": by_speaker,
            "definition": "Summed diarized turn duration; overlapping turns can exceed wall-clock speech time."}


def switching(turns: list[dict[str, Any]], duration_s: Optional[float], config: Thresholds) -> dict[str, Any]:
    sequence, pairs, matrix = [], Counter(), defaultdict(Counter)
    for turn in turns:
        if not sequence or sequence[-1] != turn["speaker"]:
            if sequence:
                pairs[(sequence[-1], turn["speaker"])] += 1
                matrix[sequence[-1]][turn["speaker"]] += 1
            sequence.append(turn["speaker"])
    total = sum(pairs.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in pairs.values()) if total else 0.0
    normalized = entropy / math.log2(len(pairs)) if len(pairs) > 1 else (0.0 if total else None)
    short = sum(turn["end"] - turn["start"] < config.short_turn_s for turn in turns)
    return {
        "total_speaker_transitions": total, "transitions_per_minute": per_minute(total, duration_s),
        "directional_matrix": {speaker: dict(sorted(targets.items())) for speaker, targets in sorted(matrix.items())},
        "transition_entropy_bits": rnd(entropy), "normalized_transition_entropy": rnd(normalized),
        "entropy_definition": "Shannon entropy over directed transition pairs; it measures diversity, not quality/fairness.",
        "short_turn_threshold_s": config.short_turn_s, "short_turn_count": short,
        "short_turn_pct": pct(short, len(turns)), "short_turns_per_minute": per_minute(short, duration_s),
    }


def validate_overlaps(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], [issue("missing_or_invalid_overlap_candidates", "overlap_candidates is not a list")]
    events, bad = [], 0
    for row in raw:
        if not isinstance(row, dict):
            bad += 1
            continue
        start, end, speakers = numeric(row.get("start")), numeric(row.get("end")), row.get("speakers")
        if start is None or end is None or end <= start or not isinstance(speakers, list):
            bad += 1
        else:
            events.append({"start": start, "end": end, "duration": end - start,
                           "speakers": sorted({item for item in speakers if isinstance(item, str) and item}),
                           "case": row.get("case"), "source": row.get("source")})
    warnings = [issue("invalid_overlap_candidate_excluded", "Invalid timing or speakers", bad)] if bad else []
    events.sort(key=lambda event: (event["start"], event["end"], str(event["case"])))
    unique = {(event["start"], event["end"], tuple(event["speakers"]), event["case"], event["source"]) for event in events}
    if len(unique) != len(events):
        warnings.append(issue("duplicate_overlap_candidates", "Source events stay counted; temporal union deduplicates them", len(events) - len(unique)))
    return events, warnings


def union_duration(events: list[dict[str, Any]]) -> float:
    """Unique elapsed time: duplicate, nested, and adjacent intervals are unioned."""
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


def overlap_metrics(events: list[dict[str, Any]], duration_s: Optional[float]) -> dict[str, Any]:
    cumulative, unique = sum(event["duration"] for event in events), union_duration(events)
    return {
        "status": "candidate_events",
        "event_definition": "Detector-reconciliation candidates, not confirmed or manually validated overlap events.",
        "candidate_event_count": len(events), "candidate_events_per_minute": per_minute(len(events), duration_s),
        "unique_elapsed_overlap_s": rnd(unique), "cumulative_candidate_overlap_s": rnd(cumulative),
        "overlap_ratio_pct": pct(unique, duration_s),
        "mean_candidate_duration_s": rnd(cumulative / len(events)) if events else None,
        "max_candidate_duration_s": rnd(max((event["duration"] for event in events), default=None)),
        "candidate_cases": dict(sorted(Counter(str(event["case"]) for event in events).items())),
        "duplicate_nested_handling": "Source count is preserved; elapsed overlap is the temporal union.",
    }


def validate_interruptions(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], [issue("missing_or_invalid_interruptions", "interruptions is not a list")]
    events, bad = [], 0
    for row in raw:
        if not isinstance(row, dict):
            bad += 1
            continue
        start, end, speaker, speakers = numeric(row.get("start")), numeric(row.get("end")), row.get("interrupter"), row.get("speakers")
        victims = sorted({item for item in speakers if isinstance(item, str) and item != speaker}) if isinstance(speakers, list) else []
        if start is None or end is None or end <= start or not isinstance(speaker, str) or not victims:
            bad += 1
        else:
            events.append({"start": start, "end": end, "duration": end - start, "interrupter": speaker, "victims": victims})
    return sorted(events, key=lambda event: (event["start"], event["end"], event["interrupter"])), [issue("invalid_interruption_excluded", "Missing timing, interrupter, or victim", bad)] if bad else []


def floor_outcome(event: dict[str, Any], turns: list[dict[str, Any]], config: Thresholds) -> str:
    """Classify only when one speaker demonstrably holds the post-overlap floor."""
    deadline = event["end"] + config.floor_outcome_window_s
    holders = {turn["speaker"] for turn in turns if turn["start"] <= deadline and turn["end"] >= event["end"] + config.floor_hold_s}
    if len(holders) != 1:
        return "ambiguous"
    holder = next(iter(holders))
    return "successful" if holder == event["interrupter"] else "failed" if holder in event["victims"] else "ambiguous"


def interruption_metrics(events: list[dict[str, Any]], turns: list[dict[str, Any]], duration_s: Optional[float], config: Thresholds) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix, outcomes, classified = defaultdict(Counter), Counter(), []
    for event in events:
        outcome = floor_outcome(event, turns, config)
        outcomes[outcome] += 1
        for victim in event["victims"]:
            matrix[event["interrupter"]][victim] += 1
        classified.append({**event, "floor_outcome": outcome, "provenance": "rule_based"})
    eligible = outcomes["successful"] + outcomes["failed"]
    return {
        "status": "osd_derived_attempts",
        "event_definition": "Valid OSD-derived interruption attempts, not human-validated interruptions.",
        "total_attempts": len(events), "attempts_per_minute": per_minute(len(events), duration_s),
        "directional_matrix": {speaker: dict(sorted(targets.items())) for speaker, targets in sorted(matrix.items())},
        "multiple_victim_rule": "An event creates one directional edge for each distinct non-interrupter; attempt total is not multiplied.",
        "floor_outcomes": dict(sorted(outcomes.items())), "eligible_floor_outcomes": eligible,
        "success_rate_pct": pct(outcomes["successful"], eligible), "failed_rate_pct": pct(outcomes["failed"], eligible),
        "unavailable_reason": None if eligible else "No unambiguous post-overlap floor holder under the configured rule.",
    }, classified


def is_acknowledgement(data: dict[str, Any], speaker: str, start: float, end: float) -> bool:
    words = []
    for segment in data.get("segments", []) if isinstance(data.get("segments"), list) else []:
        for word in segment.get("words", []) if isinstance(segment, dict) and isinstance(segment.get("words"), list) else []:
            left, right = numeric(word.get("start")) if isinstance(word, dict) else None, numeric(word.get("end")) if isinstance(word, dict) else None
            if isinstance(word, dict) and word.get("speaker") == speaker and left is not None and right is not None and right > start and left < end and isinstance(word.get("word"), str):
                words.append(word["word"].strip(" .,!?।").lower())
    return bool(words) and all(word in ACKS for word in words)


def classify_events(data: dict[str, Any], turns: list[dict[str, Any]], overlaps: list[dict[str, Any]], interruptions: list[dict[str, Any]], switch: dict[str, Any], duration_s: Optional[float], config: Thresholds) -> dict[str, Any]:
    clean = sum(left["speaker"] != right["speaker"] and 0 <= right["start"] - left["end"] <= config.clean_transfer_gap_s for left, right in zip(turns, turns[1:]))
    backchannels = competitive = prolonged = 0
    for event in overlaps:
        prolonged += event["duration"] >= config.prolonged_overlap_s
        continuing = {turn["speaker"] for turn in turns if turn["start"] < event["end"] and turn["end"] >= event["end"] + config.floor_hold_s}
        competitive += event["duration"] >= config.brief_backchannel_s and len(continuing.intersection(event["speakers"])) >= 2
        if any(is_acknowledgement(data, speaker, event["start"], event["end"]) for speaker in event["speakers"]):
            backchannels += 1
    outcomes = Counter(event["floor_outcome"] for event in interruptions)
    eligible = outcomes["successful"] + outcomes["failed"]
    return {
        "decision_rules": {
            "candidate": "Different-speaker chronological transition, validated overlap candidate, or OSD interruption attempt.",
            "clean_transfer": "Different speakers with a 0..clean_transfer_gap_s non-overlap gap.",
            "brief_backchannel": "Words in a small acknowledgement lexicon within an overlap; short duration alone is insufficient.",
            "competitive_overlap": "Two involved speakers continue beyond overlap end for floor_hold_s; duration alone is insufficient.",
            "interruption_outcome": "Exactly one speaker holds floor for floor_hold_s within floor_outcome_window_s; otherwise ambiguous.",
            "semantic_agreement_disagreement": "Unavailable: candidate.case=AGREE is detector corroboration, not semantic agreement.",
        },
        "thresholds": asdict(config),
        "agreement_overlap": {"count": None, "rate_pct": None, "denominator": None, "provenance": "unknown", "unavailable_reason": "No semantic agreement annotation/classifier."},
        "disagreement_overlap": {"count": None, "rate_pct": None, "denominator": None, "provenance": "unknown", "unavailable_reason": "No semantic disagreement annotation/classifier."},
        "brief_backchannel": {"count": backchannels, "rate_per_minute": per_minute(backchannels, duration_s), "denominator_overlap_candidates": len(overlaps), "provenance": "rule_based"},
        "competitive_overlap": {"count": competitive, "rate_per_minute": per_minute(competitive, duration_s), "denominator_overlap_candidates": len(overlaps), "provenance": "rule_based"},
        "prolonged_overlap": {"count": prolonged, "rate_per_minute": per_minute(prolonged, duration_s), "denominator_overlap_candidates": len(overlaps), "provenance": "rule_based"},
        "clean_turn_transfer": {"count": clean, "rate_pct_of_transitions": pct(clean, switch["total_speaker_transitions"]), "denominator_transitions": switch["total_speaker_transitions"], "provenance": "rule_based"},
        "successful_interruption": {"count": outcomes["successful"], "denominator_eligible_attempts": eligible, "provenance": "rule_based"},
        "failed_interruption": {"count": outcomes["failed"], "denominator_eligible_attempts": eligible, "rate_pct": pct(outcomes["failed"], eligible), "provenance": "rule_based"},
        "unclassified_or_ambiguous_interruption_count": outcomes["ambiguous"],
    }


def analyze_episode(directory_id: str, data: dict[str, Any], config: Thresholds) -> dict[str, Any]:
    turns, turn_warnings = validate_turns(data.get("speaker_turns"))
    overlaps, overlap_warnings = validate_overlaps(data.get("overlap_candidates"))
    attempts, interruption_warnings = validate_interruptions(data.get("interruptions"))
    duration_s, duration_source = duration(data, turns)
    merged = merge_turns(turns, config.merge_gap_s)
    switch = switching(merged, duration_s, config)
    interruption_data, classified = interruption_metrics(attempts, turns, duration_s, config)
    warnings = turn_warnings + overlap_warnings + interruption_warnings
    participants = data.get("participants")
    if not isinstance(participants, list) or not participants:
        warnings.append(issue("missing_participants", "No participant identity/role records are available"))
    if duration_s is None:
        warnings.append(issue("unknown_episode_duration", "No valid metadata, OSD duration, or turn-end proxy is available"))
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    return {
        "metadata": {"video_id": episode.get("video_id", directory_id), "directory_id": directory_id, "title": episode.get("title"), "channel": episode.get("channel"), "duration_s": rnd(duration_s), "duration_source": duration_source, "participant_count": len(participants) if isinstance(participants, list) else None},
        "participation": participation(turns, merged), "switching": switch, "overlap": overlap_metrics(overlaps, duration_s),
        "interruptions": interruption_data, "event_classification": classify_events(data, turns, overlaps, classified, switch, duration_s, config),
        "quality": {"warning_count": len(warnings), "warnings": warnings, "input_counts": {"raw_turn_records": len(data.get("speaker_turns", [])) if isinstance(data.get("speaker_turns"), list) else None, "valid_turns": len(turns), "valid_overlap_candidates": len(overlaps), "valid_interruptions": len(attempts)}},
    }


def definitions() -> dict[str, Any]:
    return {
        "field_mapping": [
            {"metric_or_concept": "episode duration", "actual_source_field": "episode.duration_s; fallback osd_stats.total_duration_s", "required_calculation": "finite positive seconds", "availability": "directly_available", "limitations": "last turn end is a fallback proxy"},
            {"metric_or_concept": "participation/switching", "actual_source_field": "speaker_turns[].speaker/start/end", "required_calculation": "validate, sort, configurable merge", "availability": "calculable", "limitations": "diarization labels need not identify people"},
            {"metric_or_concept": "overlap", "actual_source_field": "overlap_candidates[]", "required_calculation": "candidate count, durations, temporal union", "availability": "calculable", "limitations": "candidates are not confirmed; AGREE means detector corroboration"},
            {"metric_or_concept": "interruption", "actual_source_field": "interruptions[]", "required_calculation": "validate directed attempts and floor rule", "availability": "calculable", "limitations": "OSD-derived operational attempts"},
            {"metric_or_concept": "semantic agreement/disagreement", "actual_source_field": None, "required_calculation": None, "availability": "unavailable", "limitations": "no annotations or classifier output"},
        ],
        "units": {"timestamps_and_durations": "seconds", "rates": "events per minute", "percentages": "0-100", "entropy": "bits"},
    }


def aggregate(episodes: list[dict[str, Any]], skipped: list[dict[str, Any]], failed: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [item["metadata"]["duration_s"] for item in episodes]
    transitions = [item["switching"]["total_speaker_transitions"] for item in episodes]
    overlaps = [item["overlap"]["candidate_event_count"] for item in episodes]
    attempts = [item["interruptions"]["total_attempts"] for item in episodes]
    total_duration = sum(item for item in durations if item is not None)
    warnings = Counter(code["code"] for item in episodes for code in item["quality"]["warnings"])
    return {
        "processing": {"successfully_analyzed": len(episodes), "skipped": len(skipped), "failed": len(failed)},
        "duration_s": {"total": rnd(total_duration), "mean_per_episode": mean(durations), "median_per_episode": median(durations)},
        "speaker_count": {"mean_per_episode": mean([float(item["participation"]["number_of_speakers"]) for item in episodes]), "median_per_episode": median([float(item["participation"]["number_of_speakers"]) for item in episodes])},
        "transitions": {"total": sum(transitions), "mean_per_episode": mean([float(item) for item in transitions]), "pooled_per_minute": per_minute(sum(transitions), total_duration)},
        "overlap_candidates": {"total": sum(overlaps), "mean_per_episode": mean([float(item) for item in overlaps]), "pooled_per_minute": per_minute(sum(overlaps), total_duration)},
        "interruption_attempts": {"total": sum(attempts), "mean_per_episode": mean([float(item) for item in attempts]), "pooled_per_minute": per_minute(sum(attempts), total_duration), "mean_episode_rate_per_minute": mean([item["interruptions"]["attempts_per_minute"] for item in episodes])},
        "metric_availability": {"temporal_metrics": "calculable", "semantic_agreement_disagreement": "unavailable"},
        "quality_warning_counts": dict(sorted(warnings.items())),
    }


def build_report(videos_dir: Path, config: Thresholds = DEFAULTS) -> dict[str, Any]:
    loaded, skipped, failed = load_episodes(videos_dir)
    episodes = []
    for directory_id, data in loaded:
        try:
            episodes.append(analyze_episode(directory_id, data, config))
        except (KeyError, TypeError, ValueError) as exc:
            failed.append({"video_id": directory_id, "reason": "analysis_schema_error", "detail": str(exc)})
    aggregate_data = aggregate(episodes, skipped, failed)
    return {
        "report_metadata": {"schema_version": "2.0", "input_directory": str(videos_dir), "thresholds": asdict(config), "semantic_label_policy": "Never infer agreement/disagreement from overlap or candidate.case."},
        "metric_definitions": definitions(), "aggregate": aggregate_data, "episodes": episodes, "skipped": skipped, "failed": failed,
        "quality_summary": {"episodes_with_warnings": sum(bool(item["quality"]["warnings"]) for item in episodes), "warning_counts": aggregate_data["quality_warning_counts"]},
        "limitations": ["Diarization labels may not map to people.", "Overlap candidates are not human-verified.", "Interruptions are OSD-derived operational attempts.", "Agreement/disagreement require annotations or a validated classifier.", "Temporal thresholds are configurable heuristics, not validated labels."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Hindi TV discourse from v2 final.json artifacts.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/analysis/discourse_summary.json"))
    for name, default in asdict(DEFAULTS).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    values = {name: getattr(args, name) for name in asdict(DEFAULTS)}
    if any(value < 0 for value in values.values()):
        parser.error("Thresholds must be non-negative seconds.")
    report = build_report(args.videos_dir, Thresholds(**values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    aggregate_data = report["aggregate"]
    print(f"Episodes analyzed: {aggregate_data['processing']['successfully_analyzed']} | skipped: {aggregate_data['processing']['skipped']} | failed: {aggregate_data['processing']['failed']}")
    print(f"Duration: {aggregate_data['duration_s']['total']} s | interruptions: {aggregate_data['interruption_attempts']['total']} | overlap candidates: {aggregate_data['overlap_candidates']['total']}")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
