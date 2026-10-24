#!/usr/bin/env python3
"""
Analyze speaker behavior in a short window after each detected overlap.

Input:
  - A pipeline final.json containing `overlap_candidates` and `segments`, OR
  - an overlap-events JSON plus a final.json containing transcript segments.

Example:
  python analyze_overlap_followup.py \
      --input final.json \
      --host-id SPEAKER_02 \
      --output overlap_followup_analysis.json \
      --window-seconds 3

The host ID can also be read from a host-identification JSON with --host-json.
This script records observable timing/speaker patterns; it does not infer intent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_host_id(host_data: Dict[str, Any]) -> Optional[str]:
    """Support common host-identification output layouts."""
    direct_keys = ("predicted_host", "predicted_host_id", "host_id", "predicted_label")
    for key in direct_keys:
        value = host_data.get(key)
        if isinstance(value, str) and value:
            return value

    for parent_key in ("prediction", "host", "result"):
        parent = host_data.get(parent_key)
        if isinstance(parent, dict):
            for key in direct_keys:
                value = parent.get(key)
                if isinstance(value, str) and value:
                    return value

    # Candidate-list layout: rank 1 / predicted candidate.
    for key in ("top_candidates", "candidates", "host_candidates"):
        candidates = host_data.get(key)
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("is_predicted_host") is True or candidate.get("rank") == 1:
                    for key2 in ("speaker", "speaker_id", "label", "candidate"):
                        value = candidate.get(key2)
                        if isinstance(value, str) and value:
                            return value
    return None


def extract_events(data: Any) -> List[Dict[str, Any]]:
    """Find overlap events in common JSON layouts."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []

    for key in ("overlap_events", "events", "overlap_candidates", "overlaps"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    # Some exports wrap the event list.
    for parent_key in ("data", "results", "analysis"):
        parent = data.get(parent_key)
        if isinstance(parent, dict):
            found = extract_events(parent)
            if found:
                return found
    return []


def normalize_speakers(event: Dict[str, Any]) -> List[str]:
    value = event.get("speakers", event.get("overlapping_speakers", []))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return sorted({str(s) for s in value if s is not None})
    return []


def get_text_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use transcript segments for text; fall back to speaker turns if needed."""
    segments = data.get("segments")
    if isinstance(segments, list) and segments:
        source = segments
    else:
        source = data.get("speaker_turns", [])
    result = []
    for i, seg in enumerate(source if isinstance(source, list) else []):
        if not isinstance(seg, dict):
            continue
        start = as_number(seg.get("start"))
        end = as_number(seg.get("end"))
        speaker = seg.get("speaker")
        if start is None or end is None or end <= start or not speaker:
            continue
        result.append({
            "segment_index": i,
            "start": start,
            "end": end,
            "speaker": str(speaker),
            "text": seg.get("text"),
        })
    return result


def event_id(event: Dict[str, Any], index: int) -> Any:
    for key in ("event_id", "id", "overlap_id", "candidate_id"):
        if event.get(key) is not None:
            return event[key]
    return index + 1


def analyze_event(
    event: Dict[str, Any],
    index: int,
    segments: List[Dict[str, Any]],
    host_id: str,
    window_seconds: float,
) -> Dict[str, Any]:
    start = as_number(event.get("start"))
    end = as_number(event.get("end"))
    if start is None or end is None or end < start:
        return {
            "event_id": event_id(event, index),
            "error": "Missing or invalid overlap start/end timestamps.",
            "source_event": event,
        }

    window_start = end
    window_end = end + window_seconds
    overlapping_speakers = normalize_speakers(event)

    # If the event's speaker list is absent, infer participants from segments
    # intersecting the overlap interval.
    if not overlapping_speakers:
        overlapping_speakers = sorted({
            seg["speaker"] for seg in segments
            if seg["start"] < end and seg["end"] > start
        })

    host_participated = host_id in overlapping_speakers

    sequence: List[Dict[str, Any]] = []
    for seg in segments:
        # Include any segment with speech intersecting the post-overlap window.
        # This retains speech that began during the overlap but continues after it.
        intersection_start = max(seg["start"], window_start)
        intersection_end = min(seg["end"], window_end)
        if intersection_end <= intersection_start:
            continue

        sequence.append({
            "speaker": seg["speaker"],
            "segment_index": seg["segment_index"],
            "segment_start": round(seg["start"], 3),
            "segment_end": round(seg["end"], 3),
            "window_intersection_start": round(intersection_start, 3),
            "window_intersection_end": round(intersection_end, 3),
            "relative_start_s": round(intersection_start - window_start, 3),
            "relative_end_s": round(intersection_end - window_start, 3),
            "started_during_overlap": start <= seg["start"] < end,
            "continues_from_overlap": seg["start"] < end and seg["end"] > end,
            "text": seg["text"],
        })

    sequence.sort(key=lambda x: (
        x["window_intersection_start"],
        x["window_intersection_end"],
        x["segment_index"],
    ))

    host_sequence_positions = [
        i for i, item in enumerate(sequence) if item["speaker"] == host_id
    ]
    first_speaker = sequence[0]["speaker"] if sequence else None
    speakers_present = sorted({item["speaker"] for item in sequence})
    host_spoke_after = bool(host_sequence_positions)
    host_continued_from_overlap = any(
        item["speaker"] == host_id and item["continues_from_overlap"]
        for item in sequence
    )

    # Record the first different speaker whose observed speech begins after a
    # host segment begins in the ordered sequence. The full sequence remains
    # available because speakers can overlap or have adjacent segments.
    speaker_after_host = []
    for pos in host_sequence_positions:
        host_item = sequence[pos]
        next_item = next(
            (
                item for item in sequence[pos + 1:]
                if item["speaker"] != host_id
                and item["window_intersection_start"] >= host_item["window_intersection_start"]
            ),
            None,
        )
        if next_item:
            speaker_after_host.append({
                "host_segment_index": host_item["segment_index"],
                "host_segment_window_start": host_item["window_intersection_start"],
                "next_speaker": next_item["speaker"],
                "next_segment_index": next_item["segment_index"],
                "next_relative_start_s": next_item["relative_start_s"],
            })

    return {
        "event_id": event_id(event, index),
        "overlap_start": round(start, 3),
        "overlap_end": round(end, 3),
        "overlap_duration_s": round(end - start, 3),
        "overlapping_speakers": overlapping_speakers,
        "host_speaker_id": host_id,
        "host_participated_in_overlap": host_participated,
        "observation_window": {
            "start": round(window_start, 3),
            "end": round(window_end, 3),
            "duration_s": window_seconds,
        },
        "post_overlap_speaker_sequence": sequence,
        # This means first speech present in the window, including speech that
        # started during the overlap and continued after it ended.
        "first_speaker_present_in_window": first_speaker,
        "speakers_present_in_window": speakers_present,
        "host_spoke_in_window": host_spoke_after,
        "host_continued_from_overlap_into_window": host_continued_from_overlap,
        "speaker_after_host": speaker_after_host,
        "source_event_metadata": {
            key: event.get(key)
            for key in ("case", "source", "confidence", "duration")
            if key in event
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON containing overlap events (or final.json).")
    parser.add_argument("--transcript-json", type=Path,
                        help="Optional final.json to supply transcript segments when --input is overlap_events.json.")
    parser.add_argument("--host-id", help="Known host speaker label, e.g. SPEAKER_02.")
    parser.add_argument("--host-json", type=Path,
                        help="Optional host-identification JSON; used if --host-id is omitted.")
    parser.add_argument("--output", type=Path, default=Path("overlap_followup_analysis.json"))
    parser.add_argument("--window-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if args.window_seconds <= 0:
        parser.error("--window-seconds must be greater than 0.")

    input_data = load_json(args.input)
    transcript_data = load_json(args.transcript_json) if args.transcript_json else input_data

    host_id = args.host_id
    if not host_id and args.host_json:
        host_data = load_json(args.host_json)
        if isinstance(host_data, dict):
            host_id = extract_host_id(host_data)
    if not host_id:
        parser.error("Provide --host-id or a readable --host-json with a predicted host label.")

    events = extract_events(input_data)
    if not events:
        parser.error("No overlap events found. Expected a list under overlap_events, events, overlap_candidates, or overlaps.")

    segments = get_text_segments(transcript_data if isinstance(transcript_data, dict) else {})
    if not segments:
        parser.error("No valid transcript segments or speaker_turns found in the transcript JSON.")

    results = [
        analyze_event(event, i, segments, host_id, args.window_seconds)
        for i, event in enumerate(events)
    ]

    episode = transcript_data.get("episode", {}) if isinstance(transcript_data, dict) else {}
    output = {
        "schema_version": "overlap-followup-analysis-1.1",
        "method": (
            "Include every speech segment intersecting the post-overlap observation window, "
            "including segments that began during the overlap and continued afterward."
        ),
        "host_speaker_id": host_id,
        "window_seconds": args.window_seconds,
        "video": {
            "video_id": episode.get("video_id") if isinstance(episode, dict) else None,
            "title": episode.get("title") if isinstance(episode, dict) else None,
        },
        "summary": {
            "num_overlap_events": len(results),
            "events_with_host_in_overlap": sum(
                r.get("host_participated_in_overlap") is True for r in results
            ),
            "events_with_host_speaking_in_window": sum(
                r.get("host_spoke_in_window") is True for r in results
            ),
            "events_with_host_continuing_from_overlap": sum(
                r.get("host_continued_from_overlap_into_window") is True for r in results
            ),
            "events_with_host_as_first_speaker_present": sum(
                r.get("first_speaker_present_in_window") == host_id for r in results
            ),
            "events_with_no_speech_in_window": sum(
                not r.get("post_overlap_speaker_sequence") for r in results
            ),
            "first_speaker_present_counts": {
                speaker: sum(r.get("first_speaker_present_in_window") == speaker for r in results)
                for speaker in sorted({
                    r.get("first_speaker_present_in_window")
                    for r in results
                    if r.get("first_speaker_present_in_window") is not None
                })
            },
        },
        "interpretation_note": (
            "Speech continuing from during an overlap into the observation window is counted "
            "as speech present after the overlap. These timing observations do not by themselves "
            "prove that another participant deliberately allowed the speaker to continue, nor do "
            "they prove moderation intent, interruption success, agreement, or disagreement."
        ),
        "events": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Analyzed {len(results)} overlap events.")
    print(f"Host speaker: {host_id}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()