"""Step 8: validation sampler CLI.

Usage:
  python -m v2.cli.validate SAMPLE  --video-id VIDEO  (add a video's segments)
  python -m v2.cli.validate RUN     --data-dir .      (sample + generate clips + template)
  python -m v2.cli.validate MARKS   --data-dir .      (validate marks + re-compute stats)
"""

import argparse
import json
import logging
from pathlib import Path

from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, VideoLayout, build_layout
from v2.validation.sampler import (
    build_sample_artifact,
    compute_stats,
    create_all_clips,
    sample_timestamps,
)

LOGGER = logging.getLogger("v2.cli.validate")

SEED = 42


def _load_segments_for(layout: VideoLayout) -> dict:
    if layout.diarized_transcript_path.exists():
        return load_json(layout.diarized_transcript_path)
    if layout.asr_path.exists():
        return load_json(layout.asr_path)
    return {}


def _corpus_root(data_dir) -> Path:
    """Corpus-level directory that holds pool.json / clips / samples."""
    if data_dir is not None:
        return Path(data_dir)
    return Path("data") / "v2" / "videos"


def cmd_sample(args: argparse.Namespace) -> None:
    """Append a video's segments to the candidate pool and save."""
    layout = build_layout(args.video_id, args.data_dir)
    data   = _load_segments_for(layout)
    segs   = data.get("segments", [])
    corpus = _corpus_root(args.data_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    pool_path = corpus / "pool.json"
    pool = load_json(pool_path) if pool_path.exists() else {"videos": []}
    existing_ids = {v["video_id"] for v in pool["videos"]}
    if layout.video_id not in existing_ids:
        pool["videos"].append({"video_id": layout.video_id, "segments": segs})
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    LOGGER.info("Pool updated: %d videos, %d total segments.",
                len(pool["videos"]),
                sum(len(v["segments"]) for v in pool["videos"]))


def cmd_run(args: argparse.Namespace) -> None:
    """Sample timestamps across the pool, create clips, write samples.json."""
    corpus = _corpus_root(args.data_dir)
    pool_path = corpus / "pool.json"
    if not pool_path.exists():
        LOGGER.error("No pool.json found at %s. Run `validate SAMPLE` for each video first.", pool_path)
        raise SystemExit(1)
    pool = load_json(pool_path)
    samples = sample_timestamps(pool["videos"], n=args.n, seed=SEED)
    artifact = build_sample_artifact(samples, model="whisperx-medium")

    clips_dir = corpus / "clips"
    clip_paths = create_all_clips(samples, corpus, clips_dir)
    LOGGER.info("Generated %d clips under %s", len(clip_paths), clips_dir)

    samples_path = corpus / "validation_samples.json"
    stats_path   = corpus / "validation_stats.json"
    save_json(artifact, samples_path)
    save_json(compute_stats(samples), stats_path)
    LOGGER.info("Samples saved: %s", samples_path)
    LOGGER.info("Template stats (all unanswered): %s",
                json.dumps(compute_stats(samples), indent=2))


def cmd_marks(args: argparse.Namespace) -> None:
    """Re-compute validation stats from a marked samples file."""
    samples_path = args.marks_file
    if samples_path is None:
        samples_path = _corpus_root(args.data_dir) / "validation_samples.json"
    data = load_json(samples_path)
    stats = compute_stats(data["samples"])
    stats_path = samples_path.parent / "validation_stats.json"
    save_json(stats, stats_path)
    LOGGER.info("Validation stats updated: %s", json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: evidence timestamp validation sampler.")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("SAMPLE", help="Add a video's segments to the sampling pool.")
    sp.add_argument("--video-id", required=True)
    sp.add_argument("--data-dir", type=Path)

    sp = sub.add_parser("RUN", help="Sample timestamps, create clips, write samples.json.")
    sp.add_argument("--n", type=int, default=10, help="Number of samples (default 10).")
    sp.add_argument("--data-dir", type=Path)

    sp = sub.add_parser("MARKS", help="Re-compute accuracy from a marked samples.json.")
    sp.add_argument("--marks-file", type=Path, default=None,
                    help="Path to marked samples.json (default: validation_samples.json).")
    sp.add_argument("--data-dir", type=Path)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    {"SAMPLE": cmd_sample, "RUN": cmd_run, "MARKS": cmd_marks}[args.command](args)


if __name__ == "__main__":
    main()
