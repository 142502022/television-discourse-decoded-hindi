#!/usr/bin/env python3
"""Create floor_management_analysis.json from an episode's existing JSON artifacts.

Expected episode directory layout:
  EPISODE_DIR/
    asr/diarization.json
    asr/diarized_transcript.json
    host_identification/host_identification.json
    overlapped/overlap_followup_analysis.json

Example:
  python create_floor_management_analysis.py --episode-dir data/v2/videos/eiY9KJ2SHnA

The script does not modify source files. Labels are rule-based observations/candidates,
not proof of speaker intention. Manual review is required for ambiguous cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "floor-management-analysis-1.0"
# DEFAULT_WINDOW_SECONDS = 3.0
POST_OVERLAP_WINDOW_SECONDS = 3.0
DATASET_DIR = Path(
    "data/v2/videos"
)

OUTPUT_JSON_PATH = "floor_management_analysis.json"

# These patterns are deliberately conservative and should be reviewed for each language.
ALLOCATE_PATTERNS = [
    re.compile(r"\b(respond|reply|answer)\b", re.I),
    re.compile(r"\b(go ahead|your response|you want to respond)\b", re.I),
    re.compile(r"\b(take it|please speak|your turn)\b", re.I),
]
REDIRECT_PATTERNS = [
    re.compile(r"\b(respond|reply|answer)\b.{0,100}\b(to|what)\b", re.I),
    re.compile(r"\b(to|about)\b.{0,100}\b(respond|reply|answer)\b", re.I),
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at top level: {path}")
    return data


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def intersects(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def speaker_ids_for_segment(segment: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    speaker = segment.get("speaker")
    if isinstance(speaker, str) and speaker:
        ids.add(speaker)
    words = segment.get("words")
    if isinstance(words, list):
        for word in words:
            if isinstance(word, dict) and isinstance(word.get("speaker"), str):
                ids.add(word["speaker"])
    return ids


def segment_summary(index: int, segment: Dict[str, Any], host_id: str) -> Dict[str, Any]:
    ids = sorted(speaker_ids_for_segment(segment))
    return {
        "segment_index": index,
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker_id": segment.get("speaker"),
        "word_speaker_ids": ids,
        "speaker_matches_host": host_id in ids,
        "text": str(segment.get("text", "")).strip(),
    }


def get_transcript_segments(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("transcript JSON field 'segments' must be a list")
    return [s for s in segments if isinstance(s, dict)]


def host_turn_context(
    turn: Dict[str, Any],
    turn_index: int,
    host_id: str,
    transcript_segments: Sequence[Dict[str, Any]],
    diarization_turns: Sequence[Dict[str, Any]],
    overlap_events: Sequence[Dict[str, Any]],
    window_seconds: float,
) -> Dict[str, Any]:
    start = float(turn["start"])
    end = float(turn["end"])

    # Keep transcript segments that temporally intersect this host turn. Attribution can
    # differ at segment vs. word level, so preserve both in the evidence record.
    linked = []
    for idx, seg in enumerate(transcript_segments):
        s0, s1 = as_float(seg.get("start")), as_float(seg.get("end"))
        if s0 is not None and s1 is not None and intersects(start, end, s0, s1):
            linked.append((idx, seg))

    host_transcript_indices: List[int] = []
    context_summaries: List[Dict[str, Any]] = []
    host_text_parts: List[str] = []
    speaker_conflicts: List[int] = []
    for idx, seg in linked:
        summary = segment_summary(idx, seg, host_id)
        context_summaries.append(summary)
        ids = set(summary["word_speaker_ids"])
        if host_id in ids:
            host_transcript_indices.append(idx)
            text = str(seg.get("text", "")).strip()
            if text:
                host_text_parts.append(text)
            if isinstance(seg.get("speaker"), str) and seg.get("speaker") != host_id and host_id in ids:
                speaker_conflicts.append(idx)

    # Other diarization turns active at the host turn's start.
    active_others = []
    for idx, other in enumerate(diarization_turns):
        if idx == turn_index:
            continue
        os_, oe = as_float(other.get("start")), as_float(other.get("end"))
        other_id = other.get("speaker")
        if os_ is None or oe is None or not isinstance(other_id, str) or other_id == host_id:
            continue
        if os_ < start < oe:
            active_others.append({
                "speaker_id": other_id,
                "turn_index": idx,
                "start": os_,
                "end": oe,
            })

    linked_overlap_ids = []
    post_window_ids = []
    resume_window_ids = []
    for ev in overlap_events:
        ov_start = as_float(ev.get("overlap_start"))
        ov_end = as_float(ev.get("overlap_end"))
        if ov_start is None or ov_end is None:
            continue
        ev_id = ev.get("event_id")
        if intersects(start, end, ov_start, ov_end):
            linked_overlap_ids.append(ev_id)
        window = ev.get("observation_window", {})
        win_start = as_float(window.get("start")) if isinstance(window, dict) else None
        win_end = as_float(window.get("end")) if isinstance(window, dict) else None
        if win_start is None:
            win_start = ov_end
        if win_end is None:
            win_end = ov_end + window_seconds
        # Host turn overlaps any part of the post-overlap window.
        if intersects(start, end, win_start, win_end):
            post_window_ids.append(ev_id)
        # Host turn starts in the post-overlap window: a new host resumption candidate.
        if win_start <= start < win_end and start >= ov_end:
            resume_window_ids.append(ev_id)

    labels: List[str] = []
    needs_review = False
    reasons: List[str] = []
    if linked_overlap_ids:
        labels.append("HOST_SPEAKS_DURING_OVERLAP")
    if active_others:
        labels.append("HOST_INTERRUPTS")
        needs_review = True
        reasons.append("Candidate interruption: another speaker's diarization turn is active at host-turn start; review transcript/audio context.")
    if resume_window_ids:
        labels.append("HOST_RESUMES")

    text = " ".join(host_text_parts).strip()
    if text and any(p.search(text) for p in ALLOCATE_PATTERNS):
        labels.append("ALLOCATE_FLOOR")
    if text and any(p.search(text) for p in REDIRECT_PATTERNS):
        labels.append("REDIRECT_RESPONSE")

    if not host_transcript_indices:
        needs_review = True
        reasons.append("No temporally linked transcript segment contains the predicted host speaker ID.")
    if speaker_conflicts:
        needs_review = True
        reasons.append("One or more transcript segments contain mixed or conflicting speaker attribution.")
    if not linked:
        needs_review = True
        reasons.append("No transcript segment overlaps this host-attributed diarization turn.")

    return {
        "host_turn_id": f"HT{len([]) + 1:03d}",  # replaced by caller
        "source_diarization_turn_index": turn_index,
        "start": start,
        "end": end,
        "speaker_id": host_id,
        "transcript_segment_indices": host_transcript_indices,
        "context_transcript_segments": context_summaries,
        "transcript_text": text,
        "overlap_event_ids": linked_overlap_ids,
        "post_overlap_window_event_ids": post_window_ids,
        "host_resume_window_event_ids": resume_window_ids,
        "other_speakers_active_at_host_turn_start": active_others,
        "observed_labels": labels,
        "needs_review": needs_review,
        "review_reason": "; ".join(reasons) if reasons else None,
        "_speaker_conflicts": speaker_conflicts,
        "_has_any_context": bool(linked),
    }


def create_guest_silence_events(
    overlap_events: Sequence[Dict[str, Any]], host_id: str
) -> List[Dict[str, Any]]:
    events = []
    for ev in overlap_events:
        window = ev.get("observation_window", {})
        if not isinstance(window, dict):
            window = {}
        host_spoke = bool(ev.get("host_spoke_in_window", False))
        sequence = ev.get("post_overlap_speaker_sequence", [])
        guest_segments = [
            s for s in sequence
            if isinstance(s, dict) and s.get("speaker") != host_id
        ] if isinstance(sequence, list) else []
        if not host_spoke and guest_segments:
            first = guest_segments[0]
            text = " ".join(str(s.get("text", "")).strip() for s in guest_segments if s.get("text"))
            events.append({
                "event_type": "post_overlap_observation",
                "start": window.get("start", ev.get("overlap_end")),
                "end": window.get("end"),
                "labels": ["GUEST_CONTINUES_HOST_SILENT"],
                "evidence": {
                    "overlap_event_id": ev.get("event_id"),
                    "overlap_start": ev.get("overlap_start"),
                    "overlap_end": ev.get("overlap_end"),
                    "first_speaker_present_in_window": ev.get("first_speaker_present_in_window"),
                    "speakers_present_in_window": ev.get("speakers_present_in_window", []),
                    "post_overlap_speaker_sequence": sequence,
                    "transcript_text": text,
                },
                "classification_status": "observed",
                "needs_review": False,
                "review_reason": None,
            })
    return events



def main() -> int:
    dataset_dir = DATASET_DIR.expanduser().resolve()
    window_seconds = POST_OVERLAP_WINDOW_SECONDS

    if window_seconds <= 0:
        print(
            "ERROR: POST_OVERLAP_WINDOW_SECONDS must be positive.",
            file=sys.stderr,
        )
        return 1

    if not dataset_dir.is_dir():
        print(
            f"ERROR: Dataset directory not found: {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    required_files = [
        Path("asr/diarization.json"),
        Path("asr/diarized_transcript.json"),
        Path("host_identification/host_identification.json"),
        Path("overlapped/overlap_followup_analysis.json"),
    ]

    # Discover episode folders automatically.
    episode_dirs = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir()
    )

    # Ignore directories that do not resemble episode folders.
    episode_dirs = [
        episode_dir
        for episode_dir in episode_dirs
        if any(
            (episode_dir / relative_path).exists()
            for relative_path in required_files
        )
    ]

    if not episode_dirs:
        print(f"No episode folders found in: {dataset_dir}")
        return 1

    successful = 0
    failed = 0
    skipped = 0

    def process_episode(episode_dir: Path) -> bool:
        output_path = episode_dir / OUTPUT_JSON_PATH

        # Check that all required inputs exist before processing.
        missing_files = [
            str(episode_dir / relative_path)
            for relative_path in required_files
            if not (episode_dir / relative_path).is_file()
        ]

        if missing_files:
            print("  Missing required input files:")
            for missing in missing_files:
                print(f"    - {missing}")
            return False

        diarization = load_json(
            episode_dir / "asr" / "diarization.json"
        )
        transcript = load_json(
            episode_dir / "asr" / "diarized_transcript.json"
        )
        host_data = load_json(
            episode_dir / "host_identification"
            / "host_identification.json"
        )
        overlap_data = load_json(
            episode_dir / "overlapped"
            / "overlap_followup_analysis.json"
        )

        host_id = (
            host_data.get("predicted_host_speaker")
            or overlap_data.get("host_speaker_id")
        )

        if not isinstance(host_id, str) or not host_id:
            raise ValueError(
                "Could not determine host speaker ID."
            )

        turns = diarization.get("turns", [])
        if not isinstance(turns, list):
            raise ValueError(
                "diarization JSON field 'turns' must be a list."
            )

        segments = get_transcript_segments(transcript)

        overlap_events = overlap_data.get("events", [])
        if not isinstance(overlap_events, list):
            raise ValueError(
                "overlap follow-up JSON field 'events' must be a list."
            )

        # Inventory every valid diarization turn attributed to the host.
        host_turns = []

        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            if turn.get("speaker") != host_id:
                continue

            if (
                as_float(turn.get("start")) is None
                or as_float(turn.get("end")) is None
            ):
                continue

            record = host_turn_context(
                turn,
                idx,
                host_id,
                segments,
                turns,
                overlap_events,
                window_seconds,
            )

            record["host_turn_id"] = (
                f"HT{len(host_turns) + 1:03d}"
            )
            host_turns.append(record)

        floor_events = []

        for host_turn in host_turns:
            labels = list(host_turn["observed_labels"])
            text = host_turn["transcript_text"]

            cue_labels = [
                label
                for label in labels
                if label in {
                    "ALLOCATE_FLOOR",
                    "REDIRECT_RESPONSE",
                }
            ]

            # Rule-matched floor-allocation cues require review.
            if cue_labels:
                host_turn["needs_review"] = True

                cue_reason = (
                    "Floor-allocation cue matched by text rules; "
                    "verify the intended participant and response "
                    "in context."
                )

                host_turn["review_reason"] = "; ".join(
                    filter(
                        None,
                        [
                            host_turn["review_reason"],
                            cue_reason,
                        ],
                    )
                )

            if not labels:
                continue

            status = (
                "candidate"
                if host_turn["needs_review"] or cue_labels
                else "observed"
            )

            floor_events.append({
                "event_type": "host_turn_behaviour",
                "host_turn_id": host_turn["host_turn_id"],
                "start": host_turn["start"],
                "end": host_turn["end"],
                "labels": labels,
                "evidence": {
                    "transcript_segment_indices":
                        host_turn["transcript_segment_indices"],
                    "transcript_text": text,
                    "context_transcript_segments":
                        host_turn["context_transcript_segments"],
                    "overlap_event_ids":
                        host_turn["overlap_event_ids"],
                    "other_speakers_active_at_host_turn_start":
                        host_turn[
                            "other_speakers_active_at_host_turn_start"
                        ],
                },
                "classification_status": status,
                "needs_review": bool(
                    host_turn["needs_review"]
                ),
                "review_reason": host_turn["review_reason"],
            })

        # Add guest-speaking observations after overlaps.
        # These are separate from the host-turn inventory.
        floor_events.extend(
            create_guest_silence_events(
                overlap_events,
                host_id,
            )
        )

        # Assign stable IDs to the generated event records.
        for idx, event in enumerate(floor_events, 1):
            event["event_id"] = f"FME{idx:04d}"

        label_counts = Counter(
            label
            for event in floor_events
            for label in event.get("labels", [])
        )

        status_counts = Counter(
            event.get("classification_status", "unresolved")
            for event in floor_events
        )

        num_with_segments = sum(
            bool(turn["transcript_segment_indices"])
            for turn in host_turns
        )

        num_no_context = sum(
            not turn["_has_any_context"]
            for turn in host_turns
        )

        num_conflicts = sum(
            bool(turn["_speaker_conflicts"])
            for turn in host_turns
        )

        num_review = sum(
            bool(event.get("needs_review"))
            for event in floor_events
        )

        # Remove internal-only fields before writing JSON.
        for turn in host_turns:
            turn.pop("_speaker_conflicts", None)
            turn.pop("_has_any_context", None)

        issues = []

        missing_host_transcript = (
            len(host_turns) - num_with_segments
        )

        if missing_host_transcript:
            issues.append(
                f"{missing_host_transcript} host turns have no "
                "temporally linked transcript segment containing "
                "the predicted host speaker ID."
            )

        if num_no_context:
            issues.append(
                f"{num_no_context} host turns have no transcript "
                "segment overlapping their timestamps."
            )

        if num_conflicts:
            issues.append(
                f"{num_conflicts} host turns have mixed or "
                "conflicting word-level speaker attribution."
            )

        language = transcript.get("language")

        if (
            isinstance(language, str)
            and language.lower() not in {"hi", "hindi"}
        ):
            issues.append(
                f"Transcript language is recorded as {language!r}; "
                "Hindi-specific cue coverage is not established."
            )

        result = {
            "schema_version": SCHEMA_VERSION,
            "host_turns": host_turns,
            "floor_management_events": floor_events,
            "summary": {
                "num_host_turns": len(host_turns),
                "num_host_turns_with_transcript_segments":
                    num_with_segments,
                "num_host_turns_without_matching_host_transcript_segments":
                    missing_host_transcript,
                "num_host_turns_without_any_temporal_transcript_context":
                    num_no_context,
                "num_host_turns_with_speaker_attribution_conflict":
                    num_conflicts,
                "num_floor_management_events": len(floor_events),
                "event_counts_by_label":
                    dict(sorted(label_counts.items())),
                "event_counts_by_status":
                    dict(sorted(status_counts.items())),
                "num_events_needing_review": num_review,
                "num_overlap_events_input": len(overlap_events),
                "post_overlap_window_seconds": window_seconds,
            },
            "validation": {
                "status": (
                    "needs_manual_review"
                    if issues or num_review
                    else "passed_rule_checks"
                ),
                "checks": {
                    "host_id_source":
                        "host_identification/host_identification.json "
                        "(fallback: overlap_followup_analysis.json)",
                    "host_turn_inventory":
                        "All valid diarization turns matching the "
                        "selected host ID are included.",
                    "transcript_linking":
                        "Transcript segments are linked by temporal "
                        "intersection and checked for speaker attribution.",
                    "overlap_linking":
                        "Overlap events are linked by temporal "
                        "intersection, not array position.",
                    "post_overlap_window":
                        f"{window_seconds:g} seconds after overlap end.",
                    "handover_policy":
                        "No handover is automatically confirmed; "
                        "the intended target and subsequent response "
                        "require contextual verification.",
                    "behaviour_vs_intent":
                        "Timing labels are observations or candidates; "
                        "they do not establish host intention.",
                },
                "issues": issues,
            },
        }

        # Write and validate the output JSON.
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        with output_path.open("r", encoding="utf-8") as file:
            json.load(file)

        print(f"  Host ID: {host_id}")
        print(f"  Host turns: {len(host_turns)}")
        print(f"  Floor-management events: {len(floor_events)}")
        print(f"  Events needing review: {num_review}")
        print(f"  Validation: {result['validation']['status']}")
        print(f"  Output: {output_path}")

        return True

    print(f"Dataset directory: {dataset_dir}")
    print(f"Episodes discovered: {len(episode_dirs)}")

    for episode_dir in episode_dirs:
        print(f"\nProcessing episode: {episode_dir.name}")

        try:
            if process_episode(episode_dir):
                successful += 1
            else:
                failed += 1

        except (FileNotFoundError, ValueError, OSError) as exc:
            failed += 1
            print(
                f"  ERROR: {exc}",
                file=sys.stderr,
            )

        except Exception as exc:
            failed += 1
            print(
                f"  UNEXPECTED ERROR: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print("\n========== BATCH SUMMARY ==========")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    return 1 if failed else 0



if __name__ == "__main__":
    raise SystemExit(main())
