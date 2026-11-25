"""Run conservative advertisement detection independently for every variant.

This command is read-only with respect to the pipeline's analysis artifacts.
It reads each variant's ``merge_final.json`` and writes a separate result to
``data/v2/debate_only/ad_detection/<variant>/<video>/ad_detection_result.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from v2.ads.hindi_ads import MIN_ADVERTISEMENT_DURATION_SECONDS, exclusion_record

VARIANTS = ("baseline", "threshold_0.5", "threshold_1.0", "threshold_2.0", "threshold_5.0")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def speaker_statistics(data: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    valid = 0
    for turn in data.get("speaker_turns", []):
        if not isinstance(turn, dict) or not isinstance(turn.get("speaker"), str):
            continue
        try:
            start, end = float(turn["start"]), float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        valid += 1
        counts[turn["speaker"]] += 1
        durations[turn["speaker"]] += end - start
    return {"valid_speaker_turn_count": valid, "speaker_count": len(counts),
            "by_speaker": {speaker: {"turn_count": counts[speaker], "attributed_seconds": round(durations[speaker], 3)}
                           for speaker in sorted(counts)}}


def detect_one(video_id: str, variant: str, source: Path, output: Path,
               min_duration_seconds: float) -> dict[str, Any]:
    data = load_json(source)
    transcript = {"segments": data.get("segments", [])}
    record = exclusion_record(video_id, transcript, min_duration_seconds=min_duration_seconds)
    summary = record["summary"]
    result = {"schema_version": "variant-ad-detection-1.0", "video_id": video_id, "variant": variant,
              "source_file": str(source), "detection_configuration": record["advertisement_detection"],
              "speaker_statistics": speaker_statistics(data), "advertisement_regions": {
                  "candidates": record["advertisement_candidates"], "confirmed": record["excluded_advertisements"],
                  "retained": record["retained_advertisements"]}, "classification_summary": summary,
              "validation": record["validation"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2/debate_only/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2/debate_only/ad_detection"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--video-id")
    parser.add_argument("--min-duration-seconds", type=float, default=MIN_ADVERTISEMENT_DURATION_SECONDS)
    args = parser.parse_args()
    variants = (args.variant,) if args.variant else VARIANTS
    videos = [args.data_dir / args.video_id] if args.video_id else sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if args.min_duration_seconds < 0:
        parser.error("--min-duration-seconds must be non-negative")
    totals = Counter(videos_discovered=len(videos), variants_discovered=len(variants))
    for video_dir in videos:
        for variant in variants:
            source = video_dir / variant / "merge_final.json"
            output = args.output_dir / variant / video_dir.name / "ad_detection_result.json"
            if not source.is_file():
                totals["missing_input_files"] += 1
                print(f"[WARNING] missing input: {source}", file=sys.stderr)
                continue
            totals["jobs_attempted"] += 1
            try:
                summary = detect_one(video_dir.name, variant, source, output, args.min_duration_seconds)
                totals["successful_jobs"] += 1
                totals["confirmed_advertisements"] += summary["total_excluded_advertisements"]
                totals["advertisement_candidates"] += summary["total_advertisement_candidates"]
                totals["uncertain_regions"] += summary["uncertain_candidate_count"]
                print(f"[OK] {video_dir.name} / {variant}: {output}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                totals["failed_jobs"] += 1
                print(f"[ERROR] {video_dir.name} / {variant}: {exc}", file=sys.stderr)
    print("Advertisement detection summary")
    for key in ("videos_discovered", "variants_discovered", "jobs_attempted", "successful_jobs", "failed_jobs", "missing_input_files", "confirmed_advertisements", "advertisement_candidates", "uncertain_regions"):
        print(f"  {key.replace('_', ' ')}: {totals[key]}")
    print(f"  output location: {args.output_dir}")
    return 1 if totals["failed_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
