#!/usr/bin/env python3
"""Build a deterministic, LLM-free overlap follow-up artifact from final.json.

This script does not modify final.json or any existing pipeline stage.
It extracts all transcript segments intersecting the 3-second window after each
recorded overlap and leaves interpretation fields null for manual verification.
"""

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path("data/v2/videos/eiY9KJ2SHnA/final.json")
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_HOST_SPEAKER = "SPEAKER_02"  # Manually confirmed for this episode.


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

        window_start = overlap_end
        window_end = window_start + window_seconds
        events = []
        for segment in segments:
            # Include any transcript segment intersecting [window_start, window_end).
            # Keep the full source segment text/timestamps; flags show partial overlap.
            if segment["start"] < window_end and segment["end"] > window_start:
                starts_in_window = segment["start"] >= window_start and segment["start"] < window_end
                active_at_window_start = segment["start"] < window_start < segment["end"]
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
            "window_start": round(window_start, 3),
            "window_end": round(window_end, 3),
            "speech_events": events,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to episode final.json")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path; defaults beside input")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--host-speaker", default=DEFAULT_HOST_SPEAKER)
    args = parser.parse_args()

    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be greater than zero")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input.resolve()}")

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    artifact = build_artifact(data, args.input, args.window_seconds, args.host_speaker)
    output = args.output or args.input.parent / "overlap_followup_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_events = sum(len(item["speech_events"]) for item in artifact["overlap_events"])
    no_speech = sum(item["observations"]["no_speech_in_window"] for item in artifact["overlap_events"])
    host_spoke = sum(item["observations"]["host_spoke_in_window"] for item in artifact["overlap_events"])
    print(f"Episode: {artifact['source']['video_id']} — {artifact['source']['title']}")
    print(f"Overlap events processed: {artifact['source']['overlap_count_processed']}")
    print(f"Speech events extracted: {total_events}")
    print(f"Windows with host speech: {host_spoke}")
    print(f"Windows with no speech: {no_speech}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()