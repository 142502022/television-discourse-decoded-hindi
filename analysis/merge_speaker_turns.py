
#!/usr/bin/env python3
"""
Merge consecutive speaker turns belonging to the same speaker.

Input:
    data/v2/videos/<video_id>/final.json

Output:
    data/v2/videos/<video_id>/merge_final.json

Rules:
    1. Sort speaker_turns chronologically by start.
    2. Merge consecutive turns with the same speaker.
    3. No gap threshold is applied.
    4. Preserve all fields from final.json.
    5. Preserve additional fields inside speaker turns.
    6. Do not modify the original final.json.
"""

import argparse
import json
import logging
from pathlib import Path
from copy import deepcopy


LOGGER = logging.getLogger(__name__)


def merge_speaker_turns(speaker_turns):
    """
    Merge consecutive turns from the same speaker.

    For merged turns:
        start = minimum start
        end   = maximum end

    All other fields are preserved.
    """

    if not speaker_turns:
        return []

    # Sort chronologically by start time.
    sorted_turns = sorted(
        speaker_turns,
        key=lambda turn: turn["start"]
    )

    merged_turns = []

    for turn in sorted_turns:

        # Preserve every field in the original turn.
        current_turn = deepcopy(turn)

        if not merged_turns:
            merged_turns.append(current_turn)
            continue

        previous_turn = merged_turns[-1]

        # Merge only consecutive turns belonging to the same speaker.
        if previous_turn.get("speaker") == current_turn.get("speaker"):

            # Extend the temporal boundaries.
            previous_turn["start"] = min(
                previous_turn["start"],
                current_turn["start"]
            )

            previous_turn["end"] = max(
                previous_turn["end"],
                current_turn["end"]
            )

            # Preserve any additional fields that were not
            # already present in the previous turn.
            for key, value in current_turn.items():

                if key not in ("start", "end", "speaker"):
                    if key not in previous_turn:
                        previous_turn[key] = deepcopy(value)

        else:
            # Different speaker: keep this as a separate turn.
            merged_turns.append(current_turn)

    return merged_turns


def build_merge_final(final_data):
    """
    Create merge_final.json while preserving the entire
    original final.json structure.
    """

    # Deep copy ensures the original data remains unchanged.
    merged_data = deepcopy(final_data)

    original_turns = final_data.get("speaker_turns", [])

    merged_turns = merge_speaker_turns(original_turns)

    # Replace ONLY speaker_turns.
    merged_data["speaker_turns"] = merged_turns

    return merged_data


def main():

    parser = argparse.ArgumentParser(
        description="Merge consecutive same-speaker turns in final.json."
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID whose final.json should be processed."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/v2/videos"),
        help="Data root directory (default: data/v2/videos)."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed logging."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Construct paths.
    video_dir = args.data_dir / args.video_id

    input_path = video_dir / "final.json"
    output_path = video_dir / "merge_final.json"

    # Validate input.
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Load original final.json.
    with input_path.open("r", encoding="utf-8") as file:
        final_data = json.load(file)

    original_count = len(final_data.get("speaker_turns", []))

    # Merge speaker turns without modifying original data.
    merged_data = build_merge_final(final_data)

    merged_count = len(merged_data.get("speaker_turns", []))

    # Save the new JSON file.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            merged_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    LOGGER.info("Original file : %s", input_path)
    LOGGER.info("Output file   : %s", output_path)
    LOGGER.info("Original turns: %d", original_count)
    LOGGER.info("Merged turns  : %d", merged_count)
    LOGGER.info("Turns removed : %d", original_count - merged_count)
    LOGGER.info("All other final.json fields preserved.")


if __name__ == "__main__":
    main()
