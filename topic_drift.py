"""Interruption -> topic-drift association for v2 debate artifacts.

This module links observed interruption events to the BERTopic timeline produced
by ``topic_modeling.py`` and asks a narrow, falsifiable question: *after an
interruption ends, does the active topic change sooner than it would by chance?*

It is deliberately separate from ``topic_analysis.py``. That module is
descriptive (it segments the transcript into candidate topic boundaries);
this module is associational (it compares interruption-adjacent outcome windows
against non-interruption windows on the same episode).

Design constraints, made explicit in the output:

* Topic windows come from ``topic_modeling.json`` contiguous runs, which are
  coarse (~tens of seconds). A change "within 45 s" is therefore a statement
  about run boundaries, not continuous time (audit L5).
* Interruptions are the OSD-only list carried in ``final.json``; the reconciled
  ``overlap_candidates`` are intentionally *not* used here (audit L7).
* Nested/duplicate events are unioned before any window is counted, so elapsed
  time is never double counted (audit L6).
* Proximity is not causation. Every association is reported as a candidate for
  human review, with base rates and a matched control.

Model outputs are candidate topics, never observed ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "interruption-topic-drift-1.0"
VERSION = "0.1.0"
MAX_CONTEXT_GAP_S = 60.0
EXCERPT_SPAN_S = 15.0
EXCERPT_CHARS = 280
OUTCOME_WINDOWS = (45.0, 90.0)


@dataclass(frozen=True)
class DriftConfig:
    outcome_windows: tuple[float, ...] = OUTCOME_WINDOWS
    max_context_gap_s: float = MAX_CONTEXT_GAP_S
    match_seed: int = 42
    excerpt_span_s: float = EXCERPT_SPAN_S
    excerpt_chars: int = EXCERPT_CHARS


def finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def r3(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def issue(code: str, message: str, count: int = 1) -> dict[str, Any]:
    return {"code": code, "message": message, "count": count}


def identifier(prefix: str, *parts: str) -> str:
    return prefix + "_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def label(value: str) -> str:
    return str(value).replace("_", " ").strip()


def window_key(seconds: float) -> str:
    return f"{seconds:g}"


def union_intervals(intervals: list[tuple[float, float]], join_gap: float = 0.0) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent intervals so no elapsed time is counted twice."""
    ordered = sorted((s, e) for s, e in intervals if e > s)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def interval_hits(interval: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    start, end = interval
    return any(start < other_end and other_start < end for other_start, other_end in others)


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def label_maps(report: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    subtopics = report.get("subtopics") if isinstance(report.get("subtopics"), list) else []
    broad = report.get("broad_topics") if isinstance(report.get("broad_topics"), list) else []
    return ({str(item.get("subtopic_id")): label(item.get("candidate_label", "")) for item in subtopics if isinstance(item, dict)},
            {str(item.get("broad_topic_id")): label(item.get("candidate_label", "")) for item in broad if isinstance(item, dict)})


def timeline_runs(episode: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = episode.get(key)
    runs: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return runs
    for run in raw:
        if not isinstance(run, dict):
            continue
        start, end = finite(run.get("start")), finite(run.get("end"))
        if start is None or end is None or end <= start or run.get("value") is None:
            continue
        runs.append({"value": str(run["value"]), "start": start, "end": end,
                     "probability_mean": finite(run.get("probability_mean")),
                     "unit_count": run.get("unit_count"), "speakers": run.get("speakers") or [],
                     "duration_s": finite(run.get("duration_s"))})
    return sorted(runs, key=lambda item: (item["start"], item["end"]))


def active_index(runs: list[dict[str, Any]], anchor: float, max_gap: float) -> Optional[int]:
    """Index of the run governing ``anchor`` (inside it, or the last one ending near it)."""
    for index, run in enumerate(runs):
        if run["start"] <= anchor < run["end"]:
            return index
        if run["start"] > anchor:
            break
    best: Optional[int] = None
    for index, run in enumerate(runs):
        if run["end"] <= anchor:
            best = index
        else:
            break
    if best is None:
        return None
    return best if anchor - runs[best]["end"] <= max_gap else None


def boundary_after(runs: list[dict[str, Any]], anchor: float, baseline_value: str, horizon: float) -> Optional[dict[str, Any]]:
    """First run boundary strictly after ``anchor`` that differs from the baseline."""
    for run in runs:
        if run["start"] <= anchor:
            continue
        if run["start"] > anchor + horizon:
            break
        if run["value"] != baseline_value:
            return run
    return None


def boundary_within(runs: list[dict[str, Any]], start: float, end: float, baseline_value: str) -> Optional[dict[str, Any]]:
    for run in runs:
        if run["start"] <= start or run["start"] > end:
            continue
        if run["value"] != baseline_value:
            return run
    return None


def load_interruptions(final_json: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw = final_json.get("interruptions") if isinstance(final_json.get("interruptions"), list) else []
    events: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    duplicates = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        start, end = finite(item.get("start")), finite(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        key = (round(start, 3), round(end, 3))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        events.append({"start": start, "end": end, "duration_s": end - start,
                       "interrupter": item.get("interrupter"), "speakers": item.get("speakers") or []})
    return sorted(events, key=lambda item: (item["start"], item["end"])), duplicates


def excerpt(segments: list[dict[str, Any]], start: float, end: float, limit: int) -> Optional[str]:
    pieces: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        seg_start, seg_end = finite(segment.get("start")), finite(segment.get("end"))
        text = segment.get("text")
        if seg_start is None or seg_end is None or not isinstance(text, str) or not text.strip():
            continue
        if seg_start < end and start < seg_end:
            pieces.append(text.strip())
    if not pieces:
        return None
    joined = " ".join(pieces)
    return joined[:limit] + ("…" if len(joined) > limit else "")


def assess_event(event: dict[str, Any], subtopic_runs: list[dict[str, Any]], broad_runs: list[dict[str, Any]],
                 sub_labels: dict[str, str], broad_labels: dict[str, str], segments: list[dict[str, Any]],
                 config: DriftConfig, video_id: str) -> dict[str, Any]:
    baseline_index = active_index(subtopic_runs, event["start"], config.max_context_gap_s)
    baseline = subtopic_runs[baseline_index] if baseline_index is not None else None
    broad_baseline_index = active_index(broad_runs, event["start"], config.max_context_gap_s)
    broad_baseline = broad_runs[broad_baseline_index] if broad_baseline_index is not None else None

    outcomes: dict[str, Any] = {}
    for window in config.outcome_windows:
        key = window_key(window)
        outcomes[key] = {}
        for layer, runs, base_run, labels in (
            ("subtopic", subtopic_runs, baseline, sub_labels),
            ("broad", broad_runs, broad_baseline, broad_labels),
        ):
            if base_run is None:
                outcomes[key][layer] = {"change_detected": None, "change_at_s": None, "change_latency_s": None,
                                        "changed_to": None, "during_interruption_change": None}
                continue
            changed = boundary_after(runs, event["end"], base_run["value"], window)
            during = boundary_within(runs, event["start"], event["end"], base_run["value"])
            outcomes[key][layer] = {
                "change_detected": changed is not None,
                "change_at_s": r3(changed["start"]) if changed else None,
                "change_latency_s": r3(changed["start"] - event["end"]) if changed else None,
                "changed_to": ({"topic_id": changed["value"], "candidate_label": labels.get(changed["value"]),
                                "probability_mean": r3(changed["probability_mean"])} if changed else None),
                "during_interruption_change": during is not None,
            }

    return {
        "event_id": identifier("drift", video_id, f"{event['start']:.3f}", f"{event['end']:.3f}"),
        "start": r3(event["start"]), "end": r3(event["end"]), "duration_s": r3(event["duration_s"]),
        "interrupter": event.get("interrupter"), "speakers": event.get("speakers"),
        "baseline": ({"subtopic_id": baseline["value"], "candidate_label": sub_labels.get(baseline["value"]),
                      "probability_mean": r3(baseline["probability_mean"]),
                      "broad_topic_id": broad_baseline["value"] if broad_baseline else None} if baseline else None),
        "outcomes": outcomes,
        "evidence": {"before_excerpt": excerpt(segments, event["start"] - config.excerpt_span_s, event["start"], config.excerpt_chars),
                     "after_excerpt": excerpt(segments, event["end"], event["end"] + config.excerpt_span_s, config.excerpt_chars)},
        "provenance": "osd_interruptions",
        "status": "assessed" if baseline else "insufficient_context",
        "uncertainty": "candidate_association_requires_review",
    }


def change_after_anchor(runs: list[dict[str, Any]], anchor: float, window: float, max_gap: float) -> bool:
    index = active_index(runs, anchor, max_gap)
    if index is None:
        return False
    baseline_value = runs[index]["value"]
    return boundary_after(runs, anchor, baseline_value, window) is not None


def baseline_comparison(runs: list[dict[str, Any]], events: list[dict[str, Any]], duration: Optional[float],
                        window: float, config: DriftConfig) -> dict[str, Any]:
    intervals = union_intervals([(event["start"], event["end"]) for event in events], join_gap=config.max_context_gap_s)
    assessed = [event for event in events
                if active_index(runs, event["start"], config.max_context_gap_s) is not None]
    changes = sum(1 for event in assessed
                  if change_after_anchor(runs, event["end"], window, config.max_context_gap_s))

    start = runs[0]["start"] if runs else 0.0
    stop = duration if duration is not None else (runs[-1]["end"] if runs else 0.0)
    candidates: list[tuple[float, float]] = []
    cursor = start
    step = window
    while cursor + window <= stop + 1e-6:
        candidates.append((cursor, cursor + window))
        cursor += step
    clean = [interval for interval in candidates if not interval_hits(interval, intervals)]
    base_changes = sum(1 for interval in clean
                       if change_after_anchor(runs, interval[0], window, config.max_context_gap_s))

    rng = random.Random(f"{config.match_seed}:{window:g}")
    sample_size = min(len(assessed), len(clean))
    control = rng.sample(clean, sample_size) if sample_size else []
    control_changes = sum(1 for interval in control
                          if change_after_anchor(runs, interval[0], window, config.max_context_gap_s))

    interruption_rate = changes / len(assessed) if assessed else None
    baseline_rate = base_changes / len(clean) if clean else None
    control_rate = control_changes / len(control) if control else None
    lift = (interruption_rate - baseline_rate) if interruption_rate is not None and baseline_rate is not None else None
    return {
        "window_seconds": window,
        "interruption": {"events_with_baseline": len(assessed), "changes": changes, "rate": r3(interruption_rate)},
        "non_interruption": {"windows": len(clean), "changes": base_changes, "rate": r3(baseline_rate)},
        "matched_control": {"windows": len(control), "changes": control_changes, "rate": r3(control_rate)},
        "lift": r3(lift),
        "interpretation": "candidate_association_requires_review",
    }


def analyze_episode(video_id: str, report: dict[str, Any], final_json: dict[str, Any], config: DriftConfig) -> dict[str, Any]:
    sub_labels, broad_labels = label_maps(report)
    episodes = {str(item.get("metadata", {}).get("video_id")): item for item in report.get("episodes", [])
                if isinstance(item, dict)} if isinstance(report.get("episodes"), list) else {}
    episode = episodes.get(video_id, {})
    metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
    duration = finite(metadata.get("duration_s"))
    subtopic_runs = timeline_runs(episode, "subtopic_timeline")
    broad_runs = timeline_runs(episode, "broad_timeline")
    segments = final_json.get("segments") if isinstance(final_json.get("segments"), list) else []
    events, duplicates = load_interruptions(final_json)
    warnings: list[dict[str, Any]] = []
    if duplicates:
        warnings.append(issue("duplicate_interruptions_removed", "Duplicate interruption intervals removed", duplicates))

    if not subtopic_runs:
        warnings.append(issue("insufficient_topic_timeline", "Episode has no usable topic timeline"))
        return {"metadata": {"video_id": video_id, "title": metadata.get("title"), "duration_s": duration},
                "status": "insufficient_input", "timeline": {"subtopic_runs": 0, "broad_runs": 0},
                "events": [], "comparisons": {}, "warnings": warnings}

    assessed = [assess_event(event, subtopic_runs, broad_runs, sub_labels, broad_labels, segments, config, video_id)
                for event in events]
    insufficient = sum(1 for item in assessed if item["status"] == "insufficient_context")
    if insufficient:
        warnings.append(issue("insufficient_context", "Interruption precedes the topic timeline", insufficient))
    comparisons = {window_key(window): {"subtopic": baseline_comparison(subtopic_runs, events, duration, window, config),
                                        "broad": baseline_comparison(broad_runs, events, duration, window, config)}
                   for window in config.outcome_windows}
    warnings.append(issue("window_coarseness", "Topic changes are measured at run boundaries (~tens of seconds), not continuously"))
    warnings.append(issue("proximity_not_causation", "Reported associations are candidates for human review only"))
    return {
        "metadata": {"video_id": video_id, "title": metadata.get("title"), "duration_s": duration},
        "status": "analyzed",
        "timeline": {"subtopic_runs": len(subtopic_runs), "broad_runs": len(broad_runs),
                     "coverage_start_s": r3(subtopic_runs[0]["start"]), "coverage_end_s": r3(subtopic_runs[-1]["end"]),
                     "event_count": len(events), "assessed_events": len(events) - insufficient,
                     "insufficient_context_events": insufficient},
        "events": assessed, "comparisons": comparisons, "warnings": warnings,
    }


def build_report(videos_dir: Path, topic_modeling: Path, config: Optional[DriftConfig] = None,
                 episode_filter: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
    config = config or DriftConfig()
    report = load_report(topic_modeling)
    episodes: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    selected = sorted(path for path in videos_dir.iterdir() if path.is_dir()) if videos_dir.is_dir() else []
    if episode_filter:
        selected = [path for path in selected if path.name == episode_filter]
    if limit:
        selected = selected[:limit]
    for directory in selected:
        final_path = directory / "final.json"
        if not final_path.is_file():
            continue
        try:
            final_json = load_report(final_path)
        except (OSError, ValueError) as error:
            failed.append({"video_id": directory.name, "reason": f"unreadable_final_json: {error}"})
            continue
        episodes.append(analyze_episode(directory.name, report, final_json, config))

    analyzed = [item for item in episodes if item["status"] == "analyzed"]
    total_events = sum(item["timeline"].get("event_count", 0) for item in analyzed) if analyzed else 0
    comparison_totals: dict[str, Any] = {}
    for window in config.outcome_windows:
        for layer in ("subtopic", "broad"):
            events_n = sum(item["comparisons"][window_key(window)][layer]["interruption"]["events_with_baseline"] for item in analyzed)
            events_c = sum(item["comparisons"][window_key(window)][layer]["interruption"]["changes"] for item in analyzed)
            base_n = sum(item["comparisons"][window_key(window)][layer]["non_interruption"]["windows"] for item in analyzed)
            base_c = sum(item["comparisons"][window_key(window)][layer]["non_interruption"]["changes"] for item in analyzed)
            rate_i = events_c / events_n if events_n else None
            rate_b = base_c / base_n if base_n else None
            comparison_totals.setdefault(window_key(window), {})[layer] = {
                "interruption_rate": r3(rate_i), "non_interruption_rate": r3(rate_b),
                "lift": r3(rate_i - rate_b) if rate_i is not None and rate_b is not None else None,
                "interruption_events": events_n, "non_interruption_windows": base_n}

    warning_counts = Counter(item["code"] for episode in episodes for item in episode["warnings"])
    return {
        "schema_version": SCHEMA_VERSION,
        "report_metadata": {
            "analysis_version": VERSION,
            "analysis_scope": "full_dataset" if not episode_filter and not limit else "subset",
            "method": "Compare topic-change probability in interruption-adjacent outcome windows against non-interruption windows.",
            "timeline_source": "topic_modeling.json episodes[].subtopic_timeline / broad_timeline",
            "interruption_source": "final.json interruptions (OSD-only subset; overlap_candidates excluded)",
            "config": {**asdict(config), "outcome_windows": list(config.outcome_windows)},
            "semantic_policy": "Associations are candidates for review; topic labels and changes are model outputs, not observed facts.",
        },
        "episodes": episodes,
        "aggregate": {
            "processing": {"analyzed": len(analyzed), "insufficient_input": len(episodes) - len(analyzed), "failed": len(failed)},
            "interruption_events": total_events,
            "comparisons": comparison_totals,
            "quality_warning_counts": dict(sorted(warning_counts.items())),
        },
        "failed": failed,
        "warnings": [issue("candidate_associations_require_review", "No interruption-topic association is a validated causal claim")],
        "limitations": [
            "Topic windows are coarse (~tens of seconds); changes are discrete run boundaries (audit L5).",
            "Interruptions are the OSD-only list; reconciled overlap candidates are not included (audit L7).",
            "Nested and duplicate events are unioned before counting to avoid double counting (audit L6).",
            "Proximity is not causation; small event counts give unstable rates.",
            "Topic labels and timeline come from an English model and inherit ASR/diarization errors.",
        ],
        "review_template": {
            "instructions": "For each episode confirm: is the baseline topic correct, is the detected change a genuine debate shift, and could another cause explain it?",
            "per_event_fields": ["event_id", "baseline", "outcomes", "evidence", "reviewer_verdict", "reviewer_note"],
            "reviewer_verdict_options": ["genuine_shift", "continuation", "unclear", "not_assessable"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Associate interruptions with subsequent topic changes.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--topic-modeling", type=Path, default=Path("data/v2/topic_analysis/topic_modeling.json"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/topic_analysis/interruption_topic_drift.json"))
    parser.add_argument("--windows", type=float, nargs="+", default=list(OUTCOME_WINDOWS),
                        help="Outcome windows in seconds (default: 45 90).")
    parser.add_argument("--match-seed", type=int, default=42, help="Deterministic seed for the matched control sample.")
    parser.add_argument("--max-context-gap", type=float, default=MAX_CONTEXT_GAP_S,
                        help="Max seconds outside the topic timeline before context is deemed insufficient.")
    parser.add_argument("--episode", default=None, help="Restrict to a single video_id.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N episodes (quick runs).")
    args = parser.parse_args()

    config = DriftConfig(outcome_windows=tuple(args.windows), max_context_gap_s=args.max_context_gap,
                         match_seed=args.match_seed)
    report = build_report(args.videos_dir, args.topic_modeling, config,
                          episode_filter=args.episode, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    processing = report["aggregate"]["processing"]
    print(f"Topic drift: {report['aggregate']['interruption_events']} interruptions across "
          f"{processing['analyzed']} analyzed episode(s), {processing['failed']} failed")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
