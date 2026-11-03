
#!/usr/bin/env python3

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DATA_DIR = Path("data/v2/videos")
DEFAULT_GAP_THRESHOLD = 1.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save JSON with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", path)


def get_duration(region: Dict[str, Any]) -> float:
    """Calculate duration from start and end."""
    return float(region["end"]) - float(region["start"])


def merge_overlap_regions(
    overlap_regions: List[Dict[str, Any]],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Merge overlap regions when the gap between consecutive regions
    is less than or equal to gap_threshold.

    Original regions are not modified.

    Returns:
        merged_regions:
            List of merged overlap regions.

        statistics:
            Original event count, merged event count,
            actual overlap duration, and merged interval duration.
    """

    if not overlap_regions:
        return [], {
            "original_overlap_events": 0,
            "merged_overlap_events": 0,
            "total_overlap_duration_sec": 0.0,
            "total_merged_overlap_duration_sec": 0.0,
            "gap_threshold_sec": gap_threshold,
        }

    # Sort chronologically.
    sorted_regions = sorted(
        overlap_regions,
        key=lambda region: float(region["start"]),
    )

    # Deep copy to avoid modifying original JSON data.
    merged_regions = []

    for region in sorted_regions:
        current = deepcopy(region)

        current["start"] = float(current["start"])
        current["end"] = float(current["end"])
        # Preserve original constituent regions for downstream filtering.
        current["source_regions"] = [deepcopy(region)]

        if not merged_regions:
            merged_regions.append(current)
            continue

        previous = merged_regions[-1]

        gap = current["start"] - previous["end"]

        # Merge when gap <= 1.5 seconds.
        if gap <= gap_threshold:

            previous["start"] = min(
                previous["start"],
                current["start"],
            )

            previous["end"] = max(
                previous["end"],
                current["end"],
            )

            # Update duration if the field exists.
            if "duration" in previous:
                previous["duration"] = (
                    previous["end"] - previous["start"]
                )

            # Keep all original constituent events for downstream filtering.
            previous.setdefault("source_regions", []).extend(
                deepcopy(current.get("source_regions", [current]))
            )

            # Preserve additional information from the current region.
            # Do not overwrite existing values.
            for key, value in current.items():

                if key not in ("start", "end", "duration"):

                    if key not in previous:
                        previous[key] = deepcopy(value)

        else:
            merged_regions.append(current)

    # Calculate actual overlap duration from ORIGINAL regions.
    actual_overlap_duration = sum(
        get_duration(region)
        for region in sorted_regions
    )

    # Calculate merged interval duration INCLUDING gaps.
    merged_interval_duration = sum(
        get_duration(region)
        for region in merged_regions
    )

    statistics = {
        "original_overlap_events": len(sorted_regions),
        "merged_overlap_events": len(merged_regions),
        "total_overlap_duration_sec": round(
            actual_overlap_duration, 3
        ),
        "total_merged_overlap_duration_sec": round(
            merged_interval_duration, 3
        ),
        "gap_threshold_sec": gap_threshold,
    }

    return merged_regions, statistics


def build_merge_overlap(
    input_data: Dict[str, Any],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Create a new JSON structure containing merged overlap regions
    while preserving the original JSON structure.
    """

    output_data = deepcopy(input_data)

    osd_data = output_data.get("osd", {})

    # In this pipeline, the canonical OSD overlap events are stored at the
    # top level under "overlap_candidates" (not osd.overlap_regions).
    original_regions = output_data.get("overlap_candidates", [])
    if not isinstance(original_regions, list):
        raise ValueError("Input JSON: overlap_candidates must be a list.")

    merged_regions, statistics = merge_overlap_regions(
        original_regions,
        gap_threshold=gap_threshold,
    )

    # Replace overlap_regions only in the output copy.
    osd_data["overlap_regions"] = merged_regions

    output_data["osd"] = osd_data

    # Add a summary section.
    output_data["overlap_merging"] = statistics

    return output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge OSD overlap regions separated by a small "
            "temporal gap."
        )
    )

    parser.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="YouTube video ID",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input JSON file. Defaults to "
            "data/v2/videos/<video_id>/final.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file. Defaults to "
            "data/v2/videos/<video_id>/merge_overlap.json"
        ),
    )

    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD,
        help="Maximum gap for merging overlap regions (default: 1.5 sec)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_dir = DEFAULT_DATA_DIR / args.video_id

    input_path = (
        args.input
        if args.input
        else video_dir / "final.json"
    )

    output_path = (
        args.output
        if args.output
        else video_dir / "merge_overlap.json"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input JSON not found: {input_path}"
        )

    logger.info("Video ID: %s", args.video_id)
    logger.info("Input: %s", input_path)
    logger.info("Gap threshold: %.2f sec", args.gap_threshold)

    input_data = load_json(input_path)

    output_data = build_merge_overlap(
        input_data,
        gap_threshold=args.gap_threshold,
    )

    save_json(output_data, output_path)

    statistics = output_data["overlap_merging"]

    logger.info("========== OVERLAP MERGING SUMMARY ==========")
    logger.info(
        "Original overlap events: %d",
        statistics["original_overlap_events"],
    )
    logger.info(
        "Merged overlap events: %d",
        statistics["merged_overlap_events"],
    )
    logger.info(
        "Actual overlap duration: %.3f sec",
        statistics["total_overlap_duration_sec"],
    )
    logger.info(
        "Merged interval duration (including gaps): %.3f sec",
        statistics["total_merged_overlap_duration_sec"],
    )
    logger.info("==========================================")

    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()