#!/usr/bin/env python3
"""Build isolated baseline and threshold analysis variants for V2 episodes.

The source ``final.json`` and its original OSD artifacts are read-only.  Each
variant gets its own analysis directory, a fresh merge of its own final JSON,
and a fresh gap/transition analysis.  Network/LLM stages are intentionally not
run: they require credentials and human review, and are represented as absent
optional sources by ``final_result.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_cluster_analysis import build_artifact as build_cluster_artifact
from build_overlap_followup_analysis import build_artifact as build_followup_artifact
from build_short_result_numeric import build_short_result
from build_turn_gap_analysis import build_statistics, calculate_gaps, load_valid_overlaps, load_valid_turns, merge_turns
from final_result import build_final_result
from final_result_numeric import extract_visualization_data
from host_identification import analyze_episode
from merge_overlap_regions import build_merge_overlap
from merge_speaker_turns import build_merge_final
from preprocess_overlap import get_video_paths, process_threshold
from threshold import analyze as analyze_transitions


DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 5.0)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def gap_artifact(data: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    turns = load_valid_turns(data)
    overlaps = load_valid_overlaps(data)
    gaps = calculate_gaps(turns, overlaps)
    # Same-speaker turns are merged across clean gaps; overlap remains a hard
    # boundary.  Speech duration remains distinct from the resulting span.
    merged_turns = merge_turns(turns, gaps, merge_gap=None)
    episode = data.get("episode", {})
    return {
        "schema_version": "turn-gap-analysis-1.0",
        "episode": {
            "video_id": episode.get("video_id"),
            "title": episode.get("title"),
            "duration_s": episode.get("duration_s"),
        },
        "configuration": {"merge_gap_seconds": None, "overlap_protection": True, "same_speaker_only": True},
        "source": {"file": source_name, "original_speaker_turn_count": len(turns), "overlap_candidate_count": len(overlaps)},
        "statistics": build_statistics(turns, gaps, merged_turns),
        "original_turns": turns,
        "inter_turn_gaps": gaps,
        "merged_turns": merged_turns,
    }


def _write_host_and_followup(variant_dir: Path, merged_path: Path) -> None:
    host = analyze_episode(merged_path)
    host_path = variant_dir / "host_identification" / "host_identification.json"
    write_json(host_path, host)
    speaker = host.get("predicted_host_speaker")
    if speaker:
        followup = build_followup_artifact(read_json(merged_path), merged_path, 3.0, speaker)
        write_json(variant_dir / "overlap_followup_analysis.json", followup)
        cluster = build_cluster_artifact(read_json(merged_path), followup)
        write_json(variant_dir / "overlap_cluster_analysis.json", cluster)


def run_variant(variant_dir: Path, final_data: Dict[str, Any], label: str, overwrite: bool) -> None:
    if (variant_dir / "merge_final.json").exists() and not overwrite:
        raise FileExistsError(f"Variant exists: {variant_dir}; pass --overwrite to replace it")
    variant_dir.mkdir(parents=True, exist_ok=True)
    merged = build_merge_final(copy.deepcopy(final_data))
    merged_path = variant_dir / "merge_final.json"
    write_json(merged_path, merged)
    # Gaps are derived from the unmodified diarization timeline.  Merging is a
    # separate analytical view and must not erase genuine silence intervals.
    write_json(variant_dir / "turn_gap_analysis.json", gap_artifact(final_data, "final.json"))
    analyze_transitions(merged_path, label=label, output=variant_dir / "speaker_transition_analysis.json", merged_output=variant_dir / "merged_speaker_sections.json")
    _write_host_and_followup(variant_dir, merged_path)
    result = build_final_result(variant_dir, pipeline_path=merged_path)
    write_json(variant_dir / "final_result.json", result)
    numeric = extract_visualization_data(result, final_data.get("episode", {}).get("video_id", variant_dir.parent.name))
    write_json(variant_dir / "final_result_numeric.json", numeric)
    # ``final.json`` remains the unmerged variant source.  Downstream summary
    # generation above explicitly receives ``merge_final.json`` instead.
    if not build_short_result(variant_dir):
        raise RuntimeError(f"Could not build short numeric result for {variant_dir}")


def process_video(video_dir: Path, thresholds: Iterable[float], overwrite: bool) -> None:
    source_path = video_dir / "final.json"
    source = read_json(source_path)
    baseline_dir = video_dir / "baseline"
    if baseline_dir.exists() and any(baseline_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Baseline exists: {baseline_dir}; pass --overwrite to replace it")
    if baseline_dir.exists() and overwrite:
        shutil.rmtree(baseline_dir)
    # Baseline overlap merging is an input to threshold filtering, but remains
    # contained in baseline rather than altering the source episode directory.
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_overlap = build_merge_overlap(source)
    write_json(baseline_dir / "merge_overlap.json", baseline_overlap)
    # Baseline has no duration exclusion, but still uses the authoritative
    # merged OSD episodes.  Preserve constituent records in source_regions.
    baseline_source = copy.deepcopy(source)
    baseline_source["overlap_candidates"] = copy.deepcopy(
        baseline_overlap.get("osd", {}).get("overlap_regions", [])
    )
    baseline_source["osd"] = copy.deepcopy(baseline_overlap.get("osd", {}))
    write_json(baseline_dir / "final.json", baseline_source)
    run_variant(baseline_dir, baseline_source, "baseline", overwrite)

    paths = get_video_paths(video_dir.name, video_dir.parent)
    for threshold in thresholds:
        # preprocess_overlap independently rebuilds every branch from source
        # OSD artifacts; no threshold output is ever read as another input.
        process_threshold(
            paths,
            original_layout=__import__("v2.media.layout", fromlist=["build_layout"]).build_layout(video_dir.name, video_dir.parent),
            threshold=float(threshold),
            overwrite=overwrite,
            merge_overlap_path=baseline_dir / "merge_overlap.json",
        )
        threshold_dir = video_dir / f"threshold_{float(threshold):.1f}"
        threshold_final = read_json(threshold_dir / "final.json")
        run_variant(threshold_dir, threshold_final, f"threshold_{float(threshold):.1f}", overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v2" / "videos")
    parser.add_argument("--video-id", help="Process one video ID; default processes all source final.json files.")
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS,
                        help="Thresholds to build; pass --thresholds with no values for baseline only.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only generated baseline/threshold variant directories.")
    args = parser.parse_args()
    video_dirs = [args.data_dir / args.video_id] if args.video_id else sorted(p.parent for p in args.data_dir.glob("*/final.json"))
    for video_dir in video_dirs:
        if not (video_dir / "final.json").is_file():
            raise FileNotFoundError(f"Missing source final.json: {video_dir / 'final.json'}")
        process_video(video_dir, args.thresholds, args.overwrite)
        print(f"[OK] {video_dir.name}")


if __name__ == "__main__":
    main()
