#!/usr/bin/env python3
"""Build a compact comparison JSON for original, merged, and threshold outputs.

Example:
  python analysis/build_short_result.py --video-id eiY9KJ2SHnA

Expected layout:
  data/v2/videos/<video_id>/final.json
  data/v2/videos/<video_id>/merge_final.json
  data/v2/videos/<video_id>/merge_overlap.json
  data/v2/videos/<video_id>/threshold_0.5/final.json
  data/v2/videos/<video_id>/threshold_0.5/analysis/speaker_transition_analysis.json
  (same for threshold_1.0 and threshold_2.0)
"""
import argparse
import json
from pathlib import Path


THRESHOLDS = ("0.5", "1.0", "2.0")


def read_json(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_video_id(data, fallback):
    return data.get("episode", {}).get("video_id") or fallback


def get_overlap_records(data):
    """Return canonical overlap list, supporting both current and legacy layouts."""
    osd = data.get("osd")
    if isinstance(osd, dict) and isinstance(osd.get("overlap_regions"), list):
        return osd["overlap_regions"]
    candidates = data.get("overlap_candidates")
    if isinstance(candidates, list):
        return candidates
    return []


def interval_duration(record):
    try:
        start, end = float(record["start"]), float(record["end"])
        return max(0.0, end - start)
    except (KeyError, TypeError, ValueError):
        try:
            return max(0.0, float(record.get("duration", 0.0)))
        except (TypeError, ValueError):
            return 0.0


def sum_overlap_duration(records):
    return round(sum(interval_duration(x) for x in records), 3)


def speaker_transition_total(analysis_data, key):
    ta = analysis_data.get("transition_analysis", {})
    totals = ta.get("totals", {})
    return int(totals.get(key, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    video_dir = args.root / args.video_id
    original_path = video_dir / "final.json"
    merge_final_path = video_dir / "merge_final.json"
    merge_overlap_path = video_dir / "merge_overlap.json"

    original = read_json(original_path)
    merge_final = read_json(merge_final_path)
    merge_overlap = read_json(merge_overlap_path)

    original_turns = original.get("speaker_turns", [])
    merged_turns = merge_final.get("speaker_turns", [])
    original_overlaps = get_overlap_records(original)
    merged_overlaps = get_overlap_records(merge_overlap)

    # Original transitions are computed using the same threshold.py analysis
    # logic, from the unmerged original final.json.
    original_analysis_path = (
        video_dir / "analysis" / "speaker_transition_analysis.json"
    )
    if not original_analysis_path.exists():
        raise FileNotFoundError(
            f"{original_analysis_path} not found. Run threshold.py on original "
            f"final.json first:\n"
            f"python analysis/threshold.py --video-id {args.video_id}"
        )
    original_analysis = read_json(original_analysis_path)

    result = {
        "video_id": get_video_id(original, args.video_id),
        "original": {
            "total_timestamps": len(original_turns),
            "total_speaker_transitions": speaker_transition_total(
                original_analysis, "confirmed"
            ),
            "total_overlap_events": len(original_overlaps),
            "total_overlap_duration_sec": sum_overlap_duration(original_overlaps),
        },
        "after_merge": {
            "total_timestamps": len(merged_turns),
            "total_overlap_events": len(merged_overlaps),
            # This sums merged interval lengths (includes merged gaps).
            "total_overlap_duration_sec": sum_overlap_duration(merged_overlaps),
        },
        "threshold_comparison": {},
    }

    for threshold in THRESHOLDS:
        threshold_dir = video_dir / f"threshold_{threshold}"
        threshold_final_path = threshold_dir / "final.json"
        transition_path = (
            threshold_dir / "analysis" / "speaker_transition_analysis.json"
        )

        threshold_final = read_json(threshold_final_path)
        threshold_overlaps = get_overlap_records(threshold_final)
        transition_data = read_json(transition_path)

        result["threshold_comparison"][threshold] = {
            "total_overlap_events": len(threshold_overlaps),
            "total_overlap_duration_sec": sum_overlap_duration(
                threshold_overlaps
            ),
            "candidate_transition_count": speaker_transition_total(
                transition_data, "candidate"
            ),
            "confirmed_transition_count": speaker_transition_total(
                transition_data, "confirmed"
            ),
        }

    output_path = args.output or (video_dir / "short_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Short result saved: {output_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
