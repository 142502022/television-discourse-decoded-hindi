
#!/usr/bin/env python3
"""
Generate interruption_management.json for each video under data/v2/videos.

Expected layout:
    data/v2/videos/<video_id>/overlap_followup_analysis.json
    data/v2/videos/<video_id>/host_identification/host_identification.json

Also accepts host_identification.json directly inside the video directory.

Output:
    data/v2/videos/<video_id>/interruption_management.json

Classification rule:
- If the host did not speak in the 3-second post-overlap window:
      host_action = "not_applicable"
- If the host did speak, but the action cannot be established reliably from
  the available evidence:
      host_action = "other_or_unclear"

The script does not guess between regaining_control, allocating_floor,
allowing_guest_to_continue, or redirecting_topic. Unclear events can be reviewed
to decide whether a longer observation window is needed.

Existing video_id/event_id values are preserved. The source JSON files are not
modified. Existing output files are skipped unless --overwrite is supplied.

Run from the project root:
    python generate_interruption_management.py
    python generate_interruption_management.py --videos-root data/v2/videos
    python generate_interruption_management.py --videos-root data/v2/videos --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_VIDEOS_ROOT = Path("data/v2/videos")
INPUT_FILENAME = "*/overlap_followup_analysis.json"
OUTPUT_FILENAME = "interruption_management.json"
HOST_IDENTIFICATION_CANDIDATES = (
    Path("host_identification/host_identification.json"),
    Path("host_identification.json"),
    Path("host_identification/host_identification(1).json"),
)

ALLOWED_HOST_ACTIONS = {
    "regaining_control",
    "allocating_floor",
    "allowing_guest_to_continue",
    "redirecting_topic",
    "other_or_unclear",
    "not_applicable",
}
ALLOWED_REVIEW_STATUSES = {"pending", "needs_review", "human_verified"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def first_nonempty_string(obj: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_host_id(host_data: Any) -> str | None:
    """Extract a predicted host speaker ID from common host-identification schemas."""
    if not isinstance(host_data, dict):
        return None

    direct_keys = (
        "predicted_host_id",
        "predicted_host",
        "host_speaker_id",
        "host_id",
        "predicted_label",
    )
    value = first_nonempty_string(host_data, direct_keys)
    if value:
        return value

    for nested_key in ("prediction", "host", "result", "host_identification"):
        nested = host_data.get(nested_key)
        value = first_nonempty_string(
            nested,
            ("predicted_host_id", "predicted_host", "host_speaker_id", "host_id",
             "predicted_label", "speaker_id", "speaker", "label"),
        )
        if value:
            return value

    for list_key in ("top_candidates", "candidates", "host_candidates"):
        candidates = host_data.get(list_key)
        if not isinstance(candidates, list):
            continue

        # Prefer an explicitly marked prediction, then rank 1, then first item.
        ordered = sorted(
            enumerate(candidates),
            key=lambda pair: (
                0 if isinstance(pair[1], dict) and pair[1].get("is_predicted_host") is True else 1,
                0 if isinstance(pair[1], dict) and pair[1].get("rank") == 1 else 1,
                pair[0],
            ),
        )
        for _, candidate in ordered:
            value = first_nonempty_string(
                candidate, ("speaker_id", "speaker", "label", "candidate", "host_id")
            )
            if value:
                return value

    return None


def find_host_identification_file(video_dir: Path) -> Path | None:
    for relative_path in HOST_IDENTIFICATION_CANDIDATES:
        candidate = video_dir / relative_path
        if candidate.is_file():
            return candidate

    # Permit a different filename within the standard host_identification folder.
    host_dir = video_dir / "host_identification"
    if host_dir.is_dir():
        candidates = sorted(host_dir.glob("*.json"))
        for candidate in candidates:
            if "host" in candidate.name.lower() and "identification" in candidate.name.lower():
                return candidate

    return None


def get_video_id(source: dict[str, Any], video_dir: Path) -> str:
    video = source.get("video")
    if isinstance(video, dict) and video.get("video_id"):
        return str(video["video_id"])

    nested_source = source.get("source")
    if isinstance(nested_source, dict) and nested_source.get("video_id"):
        return str(nested_source["video_id"])

    return video_dir.name


def get_host_id(video_dir: Path) -> tuple[str | None, str | None]:
    """
    Use host_identification.json as the source of truth.
    Returns (host_id, path used). No hardcoded speaker fallback is used.
    """
    host_file = find_host_identification_file(video_dir)
    if host_file is None:
        return None, None

    host_data = load_json(host_file)
    host_id = extract_host_id(host_data)
    if host_id is None:
        raise ValueError(
            f"Could not extract predicted host speaker ID from {host_file}. "
            "Please check its JSON schema; no fallback host ID will be guessed."
        )
    return host_id, str(host_file)


def get_events(source: dict[str, Any]) -> list[dict[str, Any]]:
    events = source.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]

    # Compatibility with the earlier build_overlap_followup_analysis.py schema.
    events = source.get("overlap_events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]

    return []


def get_sequence(event: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = event.get("post_overlap_speaker_sequence", [])
    if not isinstance(sequence, list):
        return []
    return [item for item in sequence if isinstance(item, dict)]


def get_host_spoke(event: dict[str, Any], host_id: str | None) -> bool | None:
    """
    Prefer the source's explicit 3-second-window flag. If absent, derive from
    the post-overlap sequence only when a host ID is available.
    """
    explicit = event.get("host_spoke_in_window")
    if isinstance(explicit, bool):
        return explicit

    if not host_id:
        return None

    return any(item.get("speaker") == host_id for item in get_sequence(event))


def get_host_evidence(event: dict[str, Any], host_id: str | None) -> str:
    """Extract host transcript excerpts from the post-overlap window only."""
    if not host_id:
        return ""

    excerpts: list[str] = []
    seen: set[tuple[Any, str]] = set()

    for segment in get_sequence(event):
        if segment.get("speaker") != host_id:
            continue

        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        segment_index = segment.get("segment_index")
        clean_text = text.strip()
        dedupe_key = (segment_index, clean_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        start = segment.get("segment_start")
        end = segment.get("segment_end")
        if start is not None and end is not None:
            excerpts.append(f"[{start}–{end}] {clean_text}")
        else:
            excerpts.append(clean_text)

    return " | ".join(excerpts)


def get_first_speaker(event: dict[str, Any]) -> Any:
    if "first_speaker_present_in_window" in event:
        return event.get("first_speaker_present_in_window")
    if "first_speaker_in_window" in event:
        return event.get("first_speaker_in_window")

    sequence = get_sequence(event)
    if sequence:
        return sequence[0].get("speaker")
    return None


def get_overlap_duration(event: dict[str, Any]) -> Any:
    duration = event.get("overlap_duration_s")
    if duration is not None:
        return duration

    start = event.get("overlap_start")
    end = event.get("overlap_end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return round(end - start, 3)
    return None


def normalize_event(
    event: dict[str, Any],
    video_id: str,
    host_id: str | None,
) -> dict[str, Any]:
    event_id = event.get("event_id")
    if event_id is None:
        raise ValueError(
            f"Event is missing event_id in video {video_id}; refusing to invent an ID."
        )

    host_spoke = get_host_spoke(event, host_id)
    evidence = get_host_evidence(event, host_id)

    # Uncertain host behavior is explicitly marked unclear so it can be audited
    # as a possible candidate for extending the post-overlap window.
    if host_spoke is False:
        host_action = "not_applicable"
    elif host_spoke is True:
        host_action = "other_or_unclear"
    else:
        # Unknown is not treated as 'not_applicable'; missing host identity/data
        # must not be mistaken for proof that the host did not speak.
        host_action = "other_or_unclear"

    return {
        "video_id": video_id,
        "event_id": event_id,
        "overlap_duration_s": get_overlap_duration(event),
        "overlapping_speakers": (
            event.get("overlapping_speakers")
            if isinstance(event.get("overlapping_speakers"), list)
            else []
        ),
        "host_participated_in_overlap": event.get("host_participated_in_overlap"),
        "host_spoke_after_overlap": host_spoke,
        "first_speaker_after_overlap": get_first_speaker(event),
        "host_action": host_action,
        "evidence": evidence,
        "confidence": None,
        "review_status": "pending",
    }


def build_output(
    source: dict[str, Any],
    input_path: Path,
    video_dir: Path,
    host_id: str | None,
    host_file: str | None,
) -> dict[str, Any]:
    video_id = get_video_id(source, video_dir)
    source_events = get_events(source)
    records = [normalize_event(event, video_id, host_id) for event in source_events]

    event_ids = [str(record["event_id"]) for record in records]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError(
            f"Duplicate event_id values found in {input_path}; refusing to alter IDs."
        )

    host_spoke_true = sum(r["host_spoke_after_overlap"] is True for r in records)
    host_spoke_false = sum(r["host_spoke_after_overlap"] is False for r in records)
    unknown_host_speech = sum(r["host_spoke_after_overlap"] is None for r in records)
    unclear = sum(r["host_action"] == "other_or_unclear" for r in records)

    return {
        "schema_version": "interruption-management-1.1",
        "video": {
            "video_id": video_id,
            "title": (
                source.get("video", {}).get("title")
                if isinstance(source.get("video"), dict)
                else None
            ),
        },
        "host_speaker_id": host_id,
        "host_identification_file": host_file,
        "source_file": input_path.name,
        "observation_window_s": 3,
        "method": (
            "Use the dedicated host-identification artifact as the host ID source. "
            "Extract evidence from the 3-second post-overlap speaker sequence. "
            "Mark unresolved host behavior as other_or_unclear rather than guessing."
        ),
        "classification_taxonomy": {
            "host_action": [
                "regaining_control",
                "allocating_floor",
                "allowing_guest_to_continue",
                "redirecting_topic",
                "other_or_unclear",
                "not_applicable",
            ],
            "review_status": ["pending", "needs_review", "human_verified"],
        },
        "summary": {
            "num_overlap_events": len(records),
            "events_host_spoke_after_overlap": host_spoke_true,
            "events_host_did_not_speak_after_overlap": host_spoke_false,
            "events_host_speech_status_unknown": unknown_host_speech,
            "events_marked_unclear_window_review": unclear,
        },
        "interpretation_note": (
            "'not_applicable' means the host is confirmed not to speak in the "
            "3-second post-overlap window. 'other_or_unclear' means the available "
            "evidence does not establish a host action confidently; review these "
            "events to assess whether extending the window resolves the ambiguity. "
            "The first speaker field reflects the source's first speaker present "
            "in the window, which may include speech that began during the overlap."
        ),
        "events": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help=f"Root containing per-video directories (default: {DEFAULT_VIDEOS_ROOT})",
    )
    parser.add_argument(
        "--input-name",
        default=INPUT_FILENAME,
        help=f"Input filename inside each video directory (default: {INPUT_FILENAME})",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_FILENAME,
        help=f"Output filename inside each video directory (default: {OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files. Without this flag, existing outputs are skipped.",
    )
    args = parser.parse_args()

    videos_root = args.videos_root
    if not videos_root.is_dir():
        print(f"ERROR: videos root does not exist: {videos_root.resolve()}", file=sys.stderr)
        return 2

    input_paths = sorted(videos_root.glob(f"*/{args.input_name}"))
    if not input_paths:
        print(
            f"No '{args.input_name}' files found directly inside video directories under "
            f"{videos_root.resolve()}",
            file=sys.stderr,
        )
        return 1

    generated = skipped = failed = 0

    for input_path in input_paths:
        video_dir = input_path.parent
        output_path = video_dir / args.output_name

        if output_path.exists() and not args.overwrite:
            print(f"SKIP (already exists): {output_path}")
            skipped += 1
            continue

        try:
            source = load_json(input_path)
            if not isinstance(source, dict):
                raise ValueError("Input JSON root must be an object.")

            host_id, host_file = get_host_id(video_dir)
            if host_id is None:
                print(
                    f"WARNING: no host identification JSON found for {video_dir.name}; "
                    "host ID remains unknown, and host actions will be marked unclear.",
                    file=sys.stderr,
                )

            output = build_output(source, input_path, video_dir, host_id, host_file)
            write_json(output_path, output)

            print(
                f"OK: {video_dir.name}: {output['summary']['num_overlap_events']} events "
                f"-> {output_path} (host={host_id or 'unknown'})"
            )
            generated += 1
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"ERROR: {input_path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nFinished. Generated: {generated}; skipped: {skipped}; failed: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

