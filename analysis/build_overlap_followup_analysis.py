#!/usr/bin/env python3
"""Build a deterministic, LLM-free overlap follow-up artifact from final.json.

This script does not modify final.json or any existing pipeline stage.
It extracts all transcript segments intersecting the 3-second window after each
recorded overlap and leaves interpretation fields null for manual verification.
"""

import argparse
import json
from pathlib import Path

# Resolve project data directory relative to this script.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VIDEOS_DIR = PROJECT_DIR /"data" /"v2" / "videos"

DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_HOST_SPEAKER = None  # Prefer per-video host_identification.json.


def as_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def valid_segments(data):
    result = []
    for index, item in enumerate(data.get("segments", [])):
        if not isinstance(item, dict):
            continue
        start = as_number(item.get("start"))
        end = as_number(item.get("end"))
        speaker = item.get("speaker")
        text = item.get("text")
        if start is None or end is None or end <= start or not speaker or not isinstance(text, str):
            continue
        result.append({
            "speaker": str(speaker),
            "start": start,
            "end": end,
            "text": text,  # Preserve the exact source text; do not trim or rewrite.
            "_source_index": index,
        })
    return sorted(result, key=lambda x: (x["start"], x["end"], x["_source_index"]))


def build_artifact(data, input_path, window_seconds, host_speaker):
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    overlaps = data.get("overlap_candidates", [])
    segments = valid_segments(data)
    records = []

    for overlap_index, overlap in enumerate(overlaps):
        if not isinstance(overlap, dict):
            continue
        overlap_start = as_number(overlap.get("start"))
        overlap_end = as_number(overlap.get("end"))
        if overlap_start is None or overlap_end is None or overlap_end <= overlap_start:
            continue

        # Existing post-overlap window: 3 seconds after overlap ends
        window_start = overlap_end
        window_end = window_start + window_seconds

        # NEW: Pre-overlap window: 3 seconds before overlap starts
        pre_window_start = max(0.0, overlap_start - window_seconds)
        pre_window_end = overlap_start

        
        events = []
        pre_overlap_events = []

        for segment in segments:

            # ==========================================
            # PRE-OVERLAP: 3 seconds BEFORE overlap
            # ==========================================
            if (segment["start"] < pre_window_end and segment["end"] > pre_window_start):
                pre_overlap_events.append({
                    "speaker": segment["speaker"],
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "text": segment["text"],

                    "starts_in_window": (
                        segment["start"] >= pre_window_start
                        and segment["start"] < pre_window_end
                    ),

                    "active_at_window_end": (
                        segment["start"] < pre_window_end
                        and segment["end"] > pre_window_end
                    ),

                    "extends_before_window": (
                        segment["start"] < pre_window_start
                    ),

                    "utterance_form": None,
                    "conversational_function": None,
                    "target_speaker": None,
                    "classification_evidence": [],
                    "classification_confidence": None,
                    "classification_status": "pending",
                })

            # ==========================================
            # POST-OVERLAP: 3 seconds AFTER overlap
            # ==========================================
            if (segment["start"] < window_end and segment["end"] > window_start):
                starts_in_window = (
                    segment["start"] >= window_start
                    and segment["start"] < window_end
                )

                active_at_window_start = (
                    segment["start"] < window_start
                    and segment["end"] > window_start
                )

                events.append({
                    "speaker": segment["speaker"],
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "text": segment["text"],
                    "starts_in_window": starts_in_window,
                    "active_at_window_start": active_at_window_start,
                    "extends_beyond_window": segment["end"] > window_end,

                    "utterance_form": None,
                    "conversational_function": None,
                    "target_speaker": None,
                    "classification_evidence": [],
                    "classification_confidence": None,
                    "classification_status": "pending",
                })

         # Maintain chronological order
        pre_overlap_events.sort(
            key=lambda event: (
                event["start"],
                event["end"],
                event["speaker"]
            )
        )
        # Identify the speaker whose segment ends closest to the overlap start.
        preceding_segments = [
            segment for segment in segments
            if segment["end"] <= overlap_start
        ]

        immediately_preceding_speaker = None

        if preceding_segments:
            last_segment = max(
                preceding_segments,
                key=lambda segment: (
                    segment["end"],
                    segment["start"]
                )
            )

            immediately_preceding_speaker = {
                "speaker": last_segment["speaker"],
                "start": round(last_segment["start"], 3),
                "end": round(last_segment["end"], 3),
                "text": last_segment["text"],
            }
        events.sort(key=lambda event: (event["start"], event["end"], event["speaker"]))
        active_at_start = [event for event in events if event["active_at_window_start"]]
        new_entries = [event for event in events if event["starts_in_window"]]
        first_new_event = min(new_entries, key=lambda event: (event["start"], event["end"])) if new_entries else None

        # A tied start is ambiguous unless the host is the only speaker at that time.
        first_new_speaker = None
        host_spoke_first_new = None
        if first_new_event is not None:
            first_start = first_new_event["start"]
            tied = [event for event in new_entries if event["start"] == first_start]
            tied_speakers = list(dict.fromkeys(event["speaker"] for event in tied))
            first_new_speaker = tied_speakers[0] if len(tied_speakers) == 1 else None
            host_spoke_first_new = (host_speaker in tied_speakers) if len(tied_speakers) == 1 else None

        # Who is already speaking at the exact window boundary? This is distinct
        # from who begins a new segment after the overlap.
        active_speakers = list(dict.fromkeys(event["speaker"] for event in active_at_start))
        if active_speakers:
            first_speaker_in_window = active_speakers[0] if len(active_speakers) == 1 else None
            host_spoke_first_in_window = (host_speaker in active_speakers) if len(active_speakers) == 1 else None
        else:
            first_speaker_in_window = first_new_speaker
            host_spoke_first_in_window = host_spoke_first_new

        host_events = [event for event in events if event["speaker"] == host_speaker]
        records.append({
            "overlap_id": overlap_index + 1,
            "overlap": {
                "start": round(overlap_start, 3),
                "end": round(overlap_end, 3),
                "duration": round(overlap_end - overlap_start, 3),
                "speakers": overlap.get("speakers", []),
                "case": overlap.get("case"),
                "source": overlap.get("source"),
                "confidence": overlap.get("confidence"),
            },
            "window_seconds": window_seconds,

            # Existing post-overlap window
            "window_start": round(window_start, 3),
            "window_end": round(window_end, 3),
            "speech_events": events,

            # NEW: Pre-overlap window
            "pre_overlap_window": {
                "start": round(pre_window_start, 3),
                "end": round(pre_window_end, 3),
                "duration": round(
                    pre_window_end - pre_window_start, 3
                ),
            },

            "pre_overlap_speaker_sequence": pre_overlap_events,

            "immediately_preceding_speaker": (
                immediately_preceding_speaker
            ),
            "observations": {
                "speech_event_count": len(events),
                "speakers_in_window": list(dict.fromkeys(event["speaker"] for event in events)),
                "host_spoke_in_window": bool(host_events),
                "speakers_active_at_window_start": active_speakers,
                "first_speaker_in_window": first_speaker_in_window,
                "host_spoke_first_in_window": host_spoke_first_in_window,
                "first_new_speaker_after_overlap": first_new_speaker,
                "host_spoke_first_new_after_overlap": host_spoke_first_new,
                "new_speech_event_count": len(new_entries),
                "no_speech_in_window": not bool(events),
            },
            "interpretation": {
                "host_behavior": None,
                "behavior_evidence": [],
                "confidence": None,
                "rationale": None,
                "manual_review_status": "pending",
                "reviewer_notes": None,
            },
        })

    return {
        "schema_version": "overlap-followup-analysis-1.1",
        "analysis_status": "deterministic_extraction_only_classification_and_interpretation_pending",
        "language_scope": {
            "selected_scope": "English-only pilot",
            "source_asr_language": (data.get("pipeline", {}).get("asr", {}).get("language")
                                     if isinstance(data.get("pipeline"), dict) and isinstance(data.get("pipeline", {}).get("asr"), dict)
                                     else None),
            "source_language_verified_by_human": False,
            "out_of_scope_policy": "Flag non-English, mixed-language, or uncertain speech for review; do not silently classify it as English.",
        },
        "classification_taxonomy": {
            "utterance_form": ["question", "statement", "imperative", "fragment", "unclear"],
            "conversational_function": [
                "asking", "challenging", "answering", "interrupting", "yielding_floor",
                "redirecting", "backchanneling", "asserting", "unclear"
            ],
            "target_speaker_policy": "Use a speaker label only when the target is supported by transcript/context; otherwise null.",
            "host_behavior": [
                "regaining_control", "challenging_participant", "asking_for_clarification",
                "redirecting_topic", "allocating_floor", "allowing_guest_to_continue",
                "summarizing", "closing_discussion", "other", "unclear", "not_applicable"
            ],
            "inference_policy": "Observable timing/order is not by itself proof of conversational intent. Use transcript evidence and abstain when evidence is insufficient.",
        },
        "manual_verification": {
            "status": "pending",
            "recommended_initial_sample_size": min(10, len(records)),
            "review_fields": [
                "event completeness", "speaker labels", "timestamps", "transcript fidelity",
                "continuation versus new entry", "first-speaker observations"
            ],
        },
        "source": {
            "file": str(input_path),
            "video_id": episode.get("video_id"),
            "title": episode.get("title"),
            "host_speaker": host_speaker,
            "window_seconds": window_seconds,
            "overlap_count_in_source": len(overlaps),
            "overlap_count_processed": len(records),
            "transcript_segment_count": len(segments),
        },
        "overlap_events": records,
        "notes": [
            "This artifact is a separate analysis output; it does not modify final.json or any existing pipeline stage.",
            "Transcript text and source segment timestamps are preserved exactly; events may extend beyond the 3-second window and are flagged.",
            "All classification and host-behavior fields are intentionally unclassified pending manual verification and later LLM analysis.",
            "first_speaker_in_window accounts for speech already active at the window boundary; first_new_speaker_after_overlap separately records a segment beginning after the overlap.",
            "Tied or ambiguous speaker starts are represented as null rather than forcing a conclusion.",
            "Speaker labels are diarization labels, not verified real-world identities.",
        ],
    }


def extract_host_speaker(host_data):
    """Find a host speaker label in host_identification.json."""
    preferred_keys = (
        "host_speaker", "identified_host", "predicted_host",
        "predicted_host_speaker", "host", "speaker_id",
    )

    def walk(obj):
        if isinstance(obj, dict):
            for key in preferred_keys:
                value = obj.get(key)
                if isinstance(value, str) and value.strip().startswith("SPEAKER_"):
                    return value.strip()
                if isinstance(value, dict):
                    for nested_key in ("speaker", "speaker_id", "label", "id"):
                        nested_value = value.get(nested_key)
                        if isinstance(nested_value, str) and nested_value.strip().startswith("SPEAKER_"):
                            return nested_value.strip()
            for value in obj.values():
                found = walk(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found:
                    return found
        return None

    return walk(host_data)


def process_video(video_dir, window_seconds, host_override=None):
    input_path = video_dir / "final.json"

    host_path = (
        video_dir
        / "host_identification"
        / "host_identification.json"
    )

    output_path = video_dir / "overlap_followup_analysis.json"

    if not input_path.is_file():
        print(f"[SKIP] {video_dir.name}: missing final.json")
        return False

    if host_override:
        host_speaker = host_override
    elif host_path.is_file():
        try:
            with host_path.open("r", encoding="utf-8") as handle:
                host_data = json.load(handle)
            host_speaker = extract_host_speaker(host_data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[SKIP] {video_dir.name}: could not read host file ({exc})")
            return False
    else:
        host_speaker = None

    if not host_speaker:
        print(f"[SKIP] {video_dir.name}: host speaker not found in host_identification.json")
        return False

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[SKIP] {video_dir.name}: could not read final.json ({exc})")
        return False

    artifact = build_artifact(data, input_path, window_seconds, host_speaker)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_events = sum(len(item["speech_events"]) for item in artifact["overlap_events"])
    no_speech = sum(item["observations"]["no_speech_in_window"] for item in artifact["overlap_events"])
    host_spoke = sum(item["observations"]["host_spoke_in_window"] for item in artifact["overlap_events"])
    print(f"[DONE] {video_dir.name} — {artifact['source'].get('title')}")
    print(f"       Host: {host_speaker}")
    print(f"       Overlaps processed: {artifact['source']['overlap_count_processed']}")
    print(f"       Speech events: {total_events}; host speech windows: {host_spoke}; empty windows: {no_speech}")
    print(f"       Output: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS
    )
    args = parser.parse_args()

    if not VIDEOS_DIR.exists():
        raise SystemExit(f"Videos directory not found: {VIDEOS_DIR}")

    video_dirs = sorted(
        path for path in VIDEOS_DIR.iterdir()
        if path.is_dir()
    )

    for video_dir in video_dirs:
        process_video(video_dir, args.window_seconds)


if __name__ == "__main__":
    main()