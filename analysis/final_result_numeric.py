#!/usr/bin/env python3
"""
Create a compact numeric-only companion JSON for visualization.

Input:
    data/v2/videos/<video_id>/final_result.json

Output:
    data/v2/videos/<video_id>/final_result_numeric.json

The file contains numeric scalar metrics and numeric count/value mappings only.
It intentionally excludes text, transcripts, raw event objects, timestamps,
booleans, and source payloads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


INPUT_NAME = "final_result.json"
OUTPUT_NAME = "final_result_numeric.json"


def is_number(value: Any) -> bool:
    # bool is a subclass of int in Python, so explicitly exclude it.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_fields(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Recursively retain numeric scalar values.
    Lists of raw events are intentionally ignored to keep this file compact.
    """
    result: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)

            if is_number(value):
                result[path] = value
            elif isinstance(value, dict):
                result.update(numeric_fields(value, path))
            # Ignore lists: most are event-level records, IDs, or timestamps.
    return result


def extract_visualization_data(final_result: dict, video_id: str) -> dict:
    """
    Create a stable compact structure with common plotting-ready metrics.
    Retain numeric paths as a fallback for new metrics added in future schemas.
    """
    numeric = numeric_fields(final_result)

    # Frequently used top-level numeric summaries, if present.
    summary = {}
    preferred_prefixes = (
        "episode_overview.",
        "speaker_participation.",
        "speaker_transitions.",
        "overlap_analysis.summary.",
    )
    for path, value in numeric.items():
        if path.startswith(preferred_prefixes):
            summary[path] = value

    # Pull numeric speaker values from the per-speaker dictionary.
    speaker_metrics = {}
    speakers = final_result.get("speaker_participation", {}).get("speakers", {})
    if isinstance(speakers, dict):
        for speaker_id, metrics in speakers.items():
            if not isinstance(metrics, dict):
                continue
            values = {k: v for k, v in metrics.items() if is_number(v)}
            if values:
                speaker_metrics[speaker_id] = values

    # Transition matrix is count-only and directly usable for heatmaps.
    transition_matrix = final_result.get("speaker_transitions", {}).get(
        "transition_matrix", {}
    )
    if not isinstance(transition_matrix, dict):
        transition_matrix = {}
    transition_matrix = {
        str(source): {
            str(target): count
            for target, count in targets.items()
            if is_number(count)
        }
        for source, targets in transition_matrix.items()
        if isinstance(targets, dict)
    }

    # Cluster counts: extract numeric values from statistics-like structures,
    # but do not flatten event-level arrays or free-text classifications.
    cluster_data = final_result.get("interaction_clusters", {}).get("data")
    cluster_numeric = numeric_fields(cluster_data, "interaction_clusters") if isinstance(cluster_data, dict) else {}

    return {
        "schema_version": "1.0",
        "video_id": video_id,  # identifier retained as a label, not a measurement
        "numeric_summary": summary,
        "speaker_metrics": speaker_metrics,
        "speaker_transition_matrix": transition_matrix,
        "cluster_numeric_metrics": cluster_numeric,
        "all_numeric_paths": numeric,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numeric-only visualization JSON from each final_result.json."
    )
    parser.add_argument("--videos-dir", default="data/v2/videos")
    parser.add_argument("--video-id", help="Process only one video directory.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.videos_dir)
    if not root.exists():
        raise SystemExit(f"[ERROR] Videos directory not found: {root}")

    video_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.video_id:
        video_dirs = [p for p in video_dirs if p.name == args.video_id]
        if not video_dirs:
            raise SystemExit(f"[ERROR] Video ID directory not found: {root / args.video_id}")

    processed = skipped = missing = 0

    for video_dir in video_dirs:
        input_path = video_dir / INPUT_NAME
        output_path = video_dir / OUTPUT_NAME

        if not input_path.exists():
            print(f"[MISSING] {video_dir.name}: {INPUT_NAME}")
            missing += 1
            continue

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {video_dir.name}: {OUTPUT_NAME} exists (use --overwrite)")
            skipped += 1
            continue

        try:
            with input_path.open("r", encoding="utf-8") as f:
                final_result = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ERROR] {video_dir.name}: cannot read {INPUT_NAME}: {exc}")
            continue

        if not isinstance(final_result, dict):
            print(f"[ERROR] {video_dir.name}: expected a JSON object in {INPUT_NAME}")
            continue

        numeric_result = extract_visualization_data(final_result, video_dir.name)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(numeric_result, f, ensure_ascii=False, indent=2)
            # Intentionally no f.write("\\n") to avoid adding an extra trailing newline.

        print(f"[OK] {video_dir.name} -> {OUTPUT_NAME}")
        processed += 1

    print(
        f"Finished. Processed={processed}, skipped={skipped}, "
        f"missing_input={missing}, scanned={len(video_dirs)}"
    )


if __name__ == "__main__":
    main()