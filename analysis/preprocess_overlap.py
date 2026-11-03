#!/usr/bin/env python3
"""
Create duration-threshold versions of overlap artifacts and regenerate final.json.

For each threshold (default: 0.5, 1.0, 2.0 seconds), this script:
  1. Removes overlap regions whose duration is <= threshold.
  2. Filters matching timestamped records from analysis.json,
     candidates.json, and overlap_followup_analysis.json.
  3. Writes the filtered artifacts under:
       data/v2/videos/<video_id>/threshold_<threshold>/
  4. Calls the existing v2.cli.finalize.build_final() using original
     non-overlap artifacts and threshold-specific analysis/candidates.

Original source artifacts are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------
# Project imports / path bootstrap
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.cli.finalize import build_final
from v2.io import save_json
from v2.media.layout import build_layout

VIDEOS_DIR = PROJECT_ROOT / "data" / "v2" / "videos"
THRESHOLDS = (0.5, 1.0, 2.0)
TIMESTAMP_TOLERANCE = 0.01
LOGGER = logging.getLogger("preprocess_overlap")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class VideoPaths:
    video_id: str
    video_dir: Path
    overlap_path: Path
    analysis_path: Path
    candidates_path: Path
    followup_path: Path
    final_path: Path
    merge_final_path: Path
    merge_overlap_path: Path


@dataclass(frozen=True)
class ThresholdPaths:
    threshold: float
    output_dir: Path
    overlap_path: Path
    analysis_path: Path
    candidates_path: Path
    followup_path: Path
    final_path: Path


def get_video_paths(video_id: str, videos_dir: Path = VIDEOS_DIR) -> VideoPaths:
    video_dir = videos_dir / video_id
    return VideoPaths(
        video_id=video_id,
        video_dir=video_dir,
        overlap_path=video_dir / "osd" / "overlap.json",
        analysis_path=video_dir / "osd" / "analysis.json",
        candidates_path=video_dir / "osd" / "candidates.json",
        followup_path=video_dir / "overlapped" / "overlap_followup_analysis.json",
        final_path=video_dir / "final.json",
        merge_final_path=video_dir / "merge_final.json",
        merge_overlap_path=video_dir / "merge_overlap.json",
    )


def get_threshold_paths(video_paths: VideoPaths, threshold: float) -> ThresholdPaths:
    output_dir = video_paths.video_dir / f"threshold_{threshold:.1f}"
    return ThresholdPaths(
        threshold=threshold,
        output_dir=output_dir,
        overlap_path=output_dir / "osd" / "overlap.json",
        analysis_path=output_dir / "osd" / "analysis.json",
        candidates_path=output_dir / "osd" / "candidates.json",
        followup_path=output_dir / "overlapped" / "overlap_followup_analysis.json",
        final_path=output_dir / "final.json",
    )


# ---------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------

def get_start(record: Dict[str, Any]) -> Optional[float]:
    for key in ("start", "overlap_start", "start_time"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def get_end(record: Dict[str, Any]) -> Optional[float]:
    for key in ("end", "overlap_end", "end_time"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def get_duration(record: Dict[str, Any]) -> Optional[float]:
    for key in ("duration", "overlap_duration_s", "duration_seconds"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    start, end = get_start(record), get_end(record)
    if start is not None and end is not None:
        return end - start
    return None


def identity(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    start, end = get_start(record), get_end(record)
    if start is None or end is None:
        return None
    return (start, end)


def matches_any_identity(
    record: Dict[str, Any],
    removed: List[Tuple[float, float]],
) -> bool:
    current = identity(record)
    if current is None:
        return False
    start, end = current
    return any(
        abs(start - r_start) <= TIMESTAMP_TOLERANCE
        and abs(end - r_end) <= TIMESTAMP_TOLERANCE
        for r_start, r_end in removed
    )


def filter_records(
    records: List[Dict[str, Any]],
    removed: List[Tuple[float, float]],
) -> Tuple[List[Dict[str, Any]], int]:
    kept, count = [], 0
    for record in records:
        if matches_any_identity(record, removed):
            count += 1
        else:
            kept.append(record)
    return kept, count


# ---------------------------------------------------------------------
# overlap.json
# ---------------------------------------------------------------------

def filter_overlap_json(
    data: Dict[str, Any],
    removed_identities: List[Tuple[float, float]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Filter legacy overlap.json records using original constituent timestamps."""
    if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
        raise ValueError("overlap.json must contain a dictionary with a 'regions' list.")

    kept, removed = [], []
    for region in data["regions"]:
        if matches_any_identity(region, removed_identities):
            removed.append(region)
        else:
            kept.append(region)

    result = copy.deepcopy(data)
    result["regions"] = kept
    return result, removed


def get_merged_overlap_regions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract merged OSD overlap regions from merge_overlap.json."""
    osd = data.get("osd", {})
    regions = osd.get("overlap_regions", []) if isinstance(osd, dict) else []
    if not isinstance(regions, list):
        raise ValueError("merge_overlap.json: osd.overlap_regions must be a list.")
    return regions


def get_constituent_identities(region: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Get original timestamp identities represented by a merged region."""
    constituents = region.get("source_regions")
    if not isinstance(constituents, list) or not constituents:
        pair = identity(region)
        return [pair] if pair is not None else []

    identities = []
    for source_region in constituents:
        pair = identity(source_region)
        if pair is not None:
            identities.append(pair)
    return identities


def filter_merged_osd_regions(
    regions: List[Dict[str, Any]], threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[float, float]]]:
    """Filter merged OSD events by merged interval duration (end-start)."""
    kept, removed, removed_source_identities = [], [], []

    for region in regions:
        start, end = get_start(region), get_end(region)
        if start is None or end is None:
            LOGGER.warning("Merged OSD region has invalid timestamps; retaining: %r", region)
            kept.append(region)
            continue

        if (end - start) <= threshold:
            removed.append(region)
            removed_source_identities.extend(get_constituent_identities(region))
        else:
            kept.append(region)

    return kept, removed, removed_source_identities


# ---------------------------------------------------------------------
# analysis.json
# ---------------------------------------------------------------------

def filter_analysis_json(
    data: Dict[str, Any], removed: List[Tuple[float, float]]
) -> Dict[str, Any]:
    result = copy.deepcopy(data)

    for key in ("regions", "events", "interruptions"):
        records = result.get(key)
        if isinstance(records, list):
            result[key], removed_count = filter_records(records, removed)
            LOGGER.debug("analysis.%s: removed %d", key, removed_count)

    stats = result.get("stats")
    if isinstance(stats, dict):
        if "num_regions" in stats:
            stats["num_regions"] = len(result.get("regions", []))
        if "overlap_seconds" in stats:
            stats["overlap_seconds"] = sum(
                duration for region in result.get("regions", [])
                if (duration := get_duration(region)) is not None and duration > 0
            )
        if "num_interruptions" in stats:
            stats["num_interruptions"] = len(result.get("interruptions", []))
        # Update common event-count statistic keys if present.
        for key in ("num_events", "total_events"):
            if key in stats and isinstance(result.get("events"), list):
                stats[key] = len(result["events"])

    return result


# ---------------------------------------------------------------------
# candidates.json
# ---------------------------------------------------------------------

def filter_candidates_json(
    data: Dict[str, Any], removed: List[Tuple[float, float]]
) -> Dict[str, Any]:
    result = copy.deepcopy(data)
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        LOGGER.warning("candidates.json has no 'candidates' list; leaving unchanged.")
        return result

    filtered, removed_count = filter_records(candidates, removed)
    result["candidates"] = filtered
    LOGGER.debug("candidates: removed %d", removed_count)

    summary = result.get("summary")
    if isinstance(summary, dict):
        summary["num_candidates"] = len(filtered)
        case_counts: Dict[str, int] = {}
        for candidate in filtered:
            case = candidate.get("case")
            if case is not None:
                case_counts[str(case)] = case_counts.get(str(case), 0) + 1

        # Recompute only fields that already exist in the source schema.
        for key, case_name in (
            ("num_agree", "AGREE"),
            ("num_osd_only", "OSD_ONLY"),
            ("num_diarization_only", "DIARIZATION_ONLY"),
        ):
            if key in summary:
                summary[key] = case_counts.get(case_name, 0)

    return result


# ---------------------------------------------------------------------
# overlap_followup_analysis.json
# ---------------------------------------------------------------------

def filter_followup_json(
    data: Dict[str, Any], removed: List[Tuple[float, float]]
) -> Dict[str, Any]:
    result = copy.deepcopy(data)
    events = result.get("events")
    if not isinstance(events, list):
        LOGGER.warning("Follow-up JSON has no 'events' list; leaving unchanged.")
        return result

    filtered, removed_count = filter_records(events, removed)
    result["events"] = filtered
    LOGGER.debug("follow-up events: removed %d", removed_count)

    summary = result.get("summary")
    if isinstance(summary, dict):
        for key in ("num_overlap_events", "num_events", "total_events"):
            if key in summary:
                summary[key] = len(filtered)

    return result


# ---------------------------------------------------------------------
# final.json regeneration
# ---------------------------------------------------------------------

def regenerate_final_json(
    video_id: str,
    original_layout: Any,
    paths: ThresholdPaths,
    threshold: float,
    removed_count: int,
    merge_final_path: Path,
) -> Dict[str, Any]:
    """
    Reuse original metadata/ASR/diarization/identity inputs, but point the
    finalizer to threshold-specific analysis.json and candidates.json.
    """
    layout_for_finalizer = SimpleNamespace(
        video_id=video_id,
        metadata_path=original_layout.metadata_path,
        asr_path=original_layout.asr_path,
        diarization_path=original_layout.diarization_path,
        diarized_transcript_path=original_layout.diarized_transcript_path,
        linked_path=original_layout.linked_path,
        roles_path=original_layout.roles_path,
        roster_path=original_layout.roster_path,
        analysis_path=paths.analysis_path,
        candidates_path=paths.candidates_path,
        final_path=paths.final_path,
    )

    final_data = build_final(layout_for_finalizer)

    # Load merged speaker turns from merge_final.json
    merged_data = load_json(
        merge_final_path
    )

    # Replace speaker_turns with merged version
    final_data["speaker_turns"] = copy.deepcopy(
        merged_data["speaker_turns"]
    )
    # Use the merged OSD overlap regions as the threshold-specific source too.
    merge_overlap_data = load_json(
        paths.output_dir.parent / "merge_overlap.json"
    )
    merged_osd_regions = get_merged_overlap_regions(merge_overlap_data)
    kept_osd_regions, _, _ = filter_merged_osd_regions(
        merged_osd_regions, threshold
    )
    if isinstance(final_data.get("osd"), dict):
        final_data["osd"]["overlap_regions"] = copy.deepcopy(kept_osd_regions)
    else:
        final_data["osd"] = {"overlap_regions": copy.deepcopy(kept_osd_regions)}
    final_data["preprocessing"] = {
        "method": "overlap_duration_threshold_filtering",
        "overlap_threshold_seconds": threshold,
        "removed_overlap_regions": removed_count,
        "rule": f"Removed merged OSD overlap intervals with duration <= {threshold:.1f} seconds.",
        "timestamp_match_tolerance_seconds": TIMESTAMP_TOLERANCE,
    }
    save_json(final_data, paths.final_path)
    return final_data


# ---------------------------------------------------------------------
# One threshold
# ---------------------------------------------------------------------

def process_threshold(
    video_paths: VideoPaths,
    original_layout: Any,
    threshold: float,
    overwrite: bool = True,
) -> Dict[str, Any]:
    paths = get_threshold_paths(video_paths, threshold)

    required = (
        video_paths.overlap_path,
        video_paths.analysis_path,
        video_paths.candidates_path,
        video_paths.followup_path,
        video_paths.merge_final_path,
        video_paths.merge_overlap_path,
    )
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input artifact(s) missing:\\n" + "\\n".join(map(str, missing))
        )

    if paths.output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {paths.output_dir}")
        # Remove only this threshold's directory, never the source video directory.
        import shutil
        shutil.rmtree(paths.output_dir)

    # Load source artifacts plus the two merged inputs.
    original_overlap = load_json(video_paths.overlap_path)
    original_analysis = load_json(video_paths.analysis_path)
    original_candidates = load_json(video_paths.candidates_path)
    original_followup = load_json(video_paths.followup_path)
    merge_overlap_data = load_json(video_paths.merge_overlap_path)

    merged_osd_regions = get_merged_overlap_regions(merge_overlap_data)
    kept_merged_regions, removed_merged_regions, removed_identities = (
        filter_merged_osd_regions(merged_osd_regions, threshold)
    )

    # Filter records by original constituent event timestamps belonging to
    # merged OSD intervals removed by this threshold.
    filtered_overlap, removed_overlap_records = filter_overlap_json(
        original_overlap, removed_identities
    )

    filtered_analysis = filter_analysis_json(original_analysis, removed_identities)
    filtered_candidates = filter_candidates_json(original_candidates, removed_identities)
    filtered_followup = filter_followup_json(original_followup, removed_identities)

    # Write only the four threshold-specific files; no recursive source copy.
    write_json(filtered_overlap, paths.overlap_path)
    write_json(filtered_analysis, paths.analysis_path)
    write_json(filtered_candidates, paths.candidates_path)
    write_json(filtered_followup, paths.followup_path)

    final_data = regenerate_final_json(
        video_id=video_paths.video_id,
        original_layout=original_layout,
        paths=paths,
        threshold=threshold,
        removed_count=len(removed_merged_regions),
        merge_final_path=video_paths.merge_final_path,
    )

    original_regions_count = len(merged_osd_regions)
    retained_regions_count = len(kept_merged_regions)
    original_candidates_count = len(original_candidates.get("candidates", []))
    retained_candidates_count = len(filtered_candidates.get("candidates", []))

    result = {
        "video_id": video_paths.video_id,
        "threshold_seconds": threshold,
        "status": "success",
        "overlap_regions_before": original_regions_count,
        "overlap_regions_removed": len(removed_merged_regions),
        "overlap_regions_remaining": retained_regions_count,
        "candidates_before": original_candidates_count,
        "candidates_removed": original_candidates_count - retained_candidates_count,
        "candidates_remaining": retained_candidates_count,
        "interruptions_remaining": len(filtered_analysis.get("interruptions", [])),
        "final_segments": len(final_data.get("segments", [])),
        "output_dir": str(paths.output_dir),
    }

    LOGGER.info(
        "threshold %.1fs | overlap regions %d -> %d (removed %d) | "
        "candidates %d -> %d | output: %s",
        threshold,
        original_regions_count,
        retained_regions_count,
        len(removed_merged_regions),
        original_candidates_count,
        retained_candidates_count,
        paths.output_dir,
    )
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter overlap artifacts by duration and regenerate threshold-specific final.json."
    )
    parser.add_argument("--video-id", required=True, help="Video/episode ID.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=VIDEOS_DIR,
        help=f"Videos data root (default: {VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(THRESHOLDS),
        help="Duration thresholds in seconds (default: 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if a threshold output directory already exists.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    video_paths = get_video_paths(args.video_id, args.data_dir)
    if not video_paths.merge_final_path.is_file():
        raise FileNotFoundError(f"merge_final.json not found: {video_paths.merge_final_path}")
    if not video_paths.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_paths.video_dir}")

    # build_layout uses data_dir as the parent root and creates standard folders.
    original_layout = build_layout(args.video_id, args.data_dir)

    summaries = []
    for threshold in args.thresholds:
        summaries.append(
            process_threshold(
                video_paths=video_paths,
                original_layout=original_layout,
                threshold=threshold,
                overwrite=not args.no_overwrite,
            )
        )

    print("\n" + "=" * 76)
    print("THRESHOLD PREPROCESSING SUMMARY")
    print("=" * 76)
    print(f"{'Threshold':>10} | {'Overlap':>22} | {'Removed':>8} | {'Candidates':>20}")
    print("-" * 76)
    for item in summaries:
        print(
            f"{item['threshold_seconds']:>9.1f}s | "
            f"{item['overlap_regions_before']:>5} -> "
            f"{item['overlap_regions_remaining']:<5} | "
            f"{item['overlap_regions_removed']:>8} | "
            f"{item['candidates_before']:>5} -> "
            f"{item['candidates_remaining']:<5}"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()