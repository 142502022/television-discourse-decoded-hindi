
#!/usr/bin/env python3
"""
Build turn_gap_analysis.json for every video in data/v2/videos.

Reads:
    <video_dir>/final.json

Writes:
    <video_dir>/turn_gap_analysis.json

Does not modify final.json.
"""

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

VIDEOS_DIR = PROJECT_DIR / "data" / "v2" / "videos"

DEFAULT_MERGE_GAP = 1.0


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def safe_float(value):
    try:
        result = float(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def calculate_stats(values):
    """Calculate descriptive statistics for a numeric list."""

    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


# ---------------------------------------------------------
# Load diarization turns
# ---------------------------------------------------------

def load_valid_turns(data):
    turns = []

    raw_turns = data.get("speaker_turns", [])

    if not isinstance(raw_turns, list):
        return turns

    for index, item in enumerate(raw_turns):

        if not isinstance(item, dict):
            continue

        speaker = item.get("speaker")
        start = safe_float(item.get("start"))
        end = safe_float(item.get("end"))

        if (
            not speaker
            or start is None
            or end is None
            or end <= start
        ):
            continue

        turns.append({
            "turn_index": index,
            "speaker": str(speaker),
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
        })

    return sorted(
        turns,
        key=lambda x: (x["start"], x["end"])
    )


# ---------------------------------------------------------
# Load overlap candidates
# ---------------------------------------------------------

def load_valid_overlaps(data):
    overlaps = []

    raw_overlaps = data.get("overlap_candidates", [])

    if not isinstance(raw_overlaps, list):
        return overlaps

    for index, item in enumerate(raw_overlaps):

        if not isinstance(item, dict):
            continue

        start = safe_float(item.get("start"))
        end = safe_float(item.get("end"))

        if start is None or end is None or end <= start:
            continue

        overlaps.append({
            "overlap_index": index,
            "start": start,
            "end": end,
            "speakers": item.get("speakers", []),
        })

    return overlaps


# ---------------------------------------------------------
# Gap overlap checking
# ---------------------------------------------------------

def gap_has_overlap(gap_start, gap_end, overlaps):
    """
    Returns True if any overlap event intersects the gap.

    A zero-duration gap cannot contain an overlap.
    """

    if gap_end <= gap_start:
        return False

    for overlap in overlaps:

        overlap_start = overlap["start"]
        overlap_end = overlap["end"]

        if (
            overlap_start < gap_end
            and overlap_end > gap_start
        ):
            return True

    return False


# ---------------------------------------------------------
# Calculate gaps between consecutive diarization turns
# ---------------------------------------------------------

def calculate_gaps(turns, overlaps):

    gaps = []

    for index in range(1, len(turns)):

        previous = turns[index - 1]
        current = turns[index]

        gap_start = previous["end"]
        gap_end = current["start"]

        gap_duration = gap_end - gap_start

        # Negative gap means temporal overlap between turns.
        if gap_duration < 0:
            gap_type = "overlapping_turns"
            has_overlap = True
            gap_duration = 0.0

        else:
            has_overlap = gap_has_overlap(
                gap_start,
                gap_end,
                overlaps
            )

            if gap_duration == 0:
                gap_type = "zero_gap"
            elif has_overlap:
                gap_type = "gap_with_overlap_event"
            elif previous["speaker"] == current["speaker"]:
                gap_type = "same_speaker_gap"
            else:
                gap_type = "speaker_transition_gap"

        gaps.append({
            "gap_id": f"G{len(gaps) + 1:04d}",
            "previous_turn_index": previous["turn_index"],
            "next_turn_index": current["turn_index"],
            "previous_speaker": previous["speaker"],
            "next_speaker": current["speaker"],
            "start": round(gap_start, 3),
            "end": round(gap_end, 3),
            "duration": round(gap_duration, 3),
            "same_speaker": (
                previous["speaker"] == current["speaker"]
            ),
            "contains_overlap": has_overlap,
            "gap_type": gap_type,
        })

    return gaps


# ---------------------------------------------------------
# Merge eligible same-speaker turns
# ---------------------------------------------------------

def merge_turns(turns, gaps, merge_gap):

    if not turns:
        return []

    merged = []

    current = {
        "merged_turn_id": "MT0001",
        "speaker": turns[0]["speaker"],
        "start": turns[0]["start"],
        "end": turns[0]["end"],
        "source_turn_indices": [turns[0]["turn_index"]],
        "internal_gaps": [],
        "speech_duration": turns[0]["duration"],
    }

    for index in range(1, len(turns)):

        previous = turns[index - 1]
        next_turn = turns[index]
        gap = gaps[index - 1]

        can_merge = (
            previous["speaker"] == next_turn["speaker"]
            and gap["duration"] <= merge_gap
            and not gap["contains_overlap"]
            and gap["duration"] >= 0
        )

        if can_merge:

            current["end"] = next_turn["end"]

            current["source_turn_indices"].append(
                next_turn["turn_index"]
            )

            current["internal_gaps"].append({
                "gap_id": gap["gap_id"],
                "start": gap["start"],
                "end": gap["end"],
                "duration": gap["duration"],
            })

            current["speech_duration"] += next_turn["duration"]

        else:

            finalize_merged_turn(current, merged)

            current = {
                "merged_turn_id": f"MT{len(merged) + 1:04d}",
                "speaker": next_turn["speaker"],
                "start": next_turn["start"],
                "end": next_turn["end"],
                "source_turn_indices": [
                    next_turn["turn_index"]
                ],
                "internal_gaps": [],
                "speech_duration": next_turn["duration"],
            }

    finalize_merged_turn(current, merged)

    return merged


def finalize_merged_turn(turn, output):

    turn["speech_duration"] = round(
        turn["speech_duration"], 3
    )

    turn["floor_holding_duration"] = round(
        turn["end"] - turn["start"], 3
    )

    turn["largest_internal_gap"] = (
        max(
            (gap["duration"] for gap in turn["internal_gaps"]),
            default=0.0
        )
    )

    turn["largest_internal_gap"] = round(
        turn["largest_internal_gap"], 3
    )

    output.append(turn)


# ---------------------------------------------------------
# Debate-level and speaker-level statistics
# ---------------------------------------------------------

def build_statistics(turns, gaps, merged_turns):

    # All positive, non-overlapping inter-turn gaps
    valid_gaps = [
        gap for gap in gaps
        if gap["duration"] > 0
        and gap["gap_type"] != "overlapping_turns"
    ]

    same_speaker_gaps = [
        gap for gap in valid_gaps
        if gap["same_speaker"]
    ]

    different_speaker_gaps = [
        gap for gap in valid_gaps
        if not gap["same_speaker"]
    ]

    merged_durations = [
        turn["floor_holding_duration"]
        for turn in merged_turns
    ]

    debate_stats = {
        "original_turn_count": len(turns),
        "merged_turn_count": len(merged_turns),
        "turns_merged": len(turns) - len(merged_turns),

        "total_gap_statistics": calculate_stats(
            [gap["duration"] for gap in valid_gaps]
        ),

        "same_speaker_gap_statistics": calculate_stats(
            [gap["duration"] for gap in same_speaker_gaps]
        ),

        "different_speaker_gap_statistics": calculate_stats(
            [gap["duration"] for gap in different_speaker_gaps]
        ),

        "overlapping_turn_pairs": sum(
            1 for gap in gaps
            if gap["gap_type"] == "overlapping_turns"
        ),

        "gaps_with_overlap_events": sum(
            1 for gap in gaps
            if gap["contains_overlap"]
        ),

        "floor_holding_duration_statistics": calculate_stats(
            merged_durations
        ),

        "longest_floor_holding_turn": (
            max(
                merged_turns,
                key=lambda x: x["floor_holding_duration"]
            )
            if merged_turns else None
        ),
    }

    # Speaker-level statistics
    speaker_turns = defaultdict(list)
    speaker_gaps = defaultdict(list)
    speaker_merged_turns = defaultdict(list)

    for turn in turns:
        speaker_turns[turn["speaker"]].append(turn)

    for gap in valid_gaps:

        # Attribute the gap to the speaker who just finished.
        speaker_gaps[gap["previous_speaker"]].append(gap)

    for turn in merged_turns:
        speaker_merged_turns[turn["speaker"]].append(turn)

    all_speakers = sorted(
        set(speaker_turns)
        | set(speaker_gaps)
        | set(speaker_merged_turns)
    )

    speaker_stats = {}

    for speaker in all_speakers:

        original = speaker_turns[speaker]
        gaps_for_speaker = speaker_gaps[speaker]
        merged = speaker_merged_turns[speaker]

        speaker_stats[speaker] = {

            "original_turn_count": len(original),

            "merged_turn_count": len(merged),

            "turns_merged": len(original) - len(merged),

            "gap_statistics": calculate_stats(
                [gap["duration"] for gap in gaps_for_speaker]
            ),

            "same_speaker_gap_count": sum(
                1 for gap in gaps_for_speaker
                if gap["same_speaker"]
            ),

            "different_speaker_gap_count": sum(
                1 for gap in gaps_for_speaker
                if not gap["same_speaker"]
            ),

            "floor_holding_duration_statistics": calculate_stats(
                [
                    turn["floor_holding_duration"]
                    for turn in merged
                ]
            ),

            "longest_floor_holding_turn": (
                max(
                    merged,
                    key=lambda x: x["floor_holding_duration"]
                )
                if merged else None
            ),
        }

    return {
        "debate_level": debate_stats,
        "speaker_level": speaker_stats,
    }


# ---------------------------------------------------------
# Process one video
# ---------------------------------------------------------

def process_video(video_dir, merge_gap):

    input_path = video_dir / "final.json"

    output_path = video_dir / "turn_gap_analysis.json"

    if not input_path.exists():
        print(f"[SKIP] Missing final.json: {video_dir.name}")
        return False

    try:
        data = read_json(input_path)

        turns = load_valid_turns(data)
        overlaps = load_valid_overlaps(data)

        if not turns:
            print(f"[SKIP] No valid speaker turns: {video_dir.name}")
            return False

        gaps = calculate_gaps(turns, overlaps)

        merged_turns = merge_turns(
            turns,
            gaps,
            merge_gap
        )

        statistics_data = build_statistics(
            turns,
            gaps,
            merged_turns
        )

        episode = data.get("episode", {})

        artifact = {

            "schema_version": "turn-gap-analysis-1.0",

            "episode": {
                "video_id": episode.get(
                    "video_id", video_dir.name
                ),
                "title": episode.get("title"),
                "duration_s": episode.get("duration_s"),
            },

            "configuration": {
                "merge_gap_seconds": merge_gap,
                "overlap_protection": True,
                "same_speaker_only": True,
            },

            "source": {
                "file": "final.json",
                "original_speaker_turn_count": len(turns),
                "overlap_candidate_count": len(overlaps),
            },

            "statistics": statistics_data,

            "original_turns": turns,

            "inter_turn_gaps": gaps,

            "merged_turns": merged_turns,
        }

        write_json(output_path, artifact)

        print(
            f"[DONE] {video_dir.name} | "
            f"Original turns: {len(turns)} | "
            f"Merged turns: {len(merged_turns)} | "
            f"Gaps: {len(gaps)}"
        )

        return True

    except Exception as exc:
        print(f"[ERROR] {video_dir.name}: {exc}")
        return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate turn gap analysis for all videos."
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help="Maximum same-speaker gap to merge (seconds). "
             "Default: 1.0"
    )

    args = parser.parse_args()

    if args.merge_gap < 0:
        parser.error("--merge-gap must be >= 0")

    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        return

    video_dirs = sorted(
        path for path in VIDEOS_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Video folders found: {len(video_dirs)}")
    print(f"Merge threshold: {args.merge_gap} seconds")
    print("-" * 60)

    processed = 0
    skipped_or_failed = 0

    for video_dir in video_dirs:

        success = process_video(
            video_dir,
            args.merge_gap
        )

        if success:
            processed += 1
        else:
            skipped_or_failed += 1

    print("-" * 60)
    print("PROCESSING COMPLETE")
    print(f"Successfully processed: {processed}")
    print(f"Skipped/failed: {skipped_or_failed}")


if __name__ == "__main__":
    main()
