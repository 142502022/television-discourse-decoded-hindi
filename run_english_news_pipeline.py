#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.label_studio_export import build_label_studio_tasks
from src.manual_review import build_manual_review_payload
from src.media.chunking import chunk_proxy_video, slice_episode_json_for_chunks
from src.media.discovery import discover_channel_episodes, last_calendar_month
from src.media.pipeline import prepare_media
from src.pipeline_status import (
    mark_episode,
    mark_stage,
    new_manifest,
    write_manifest,
)
from src.speech.final_json import build_episode_json
from src.speech.io import save_json
from src.speech.overlap import run_pyannote_osd
from src.speech.utils import diarization_intersections, merge_overlap_candidates
from src.speech.whisperx_pipeline import run_whisperx

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run English news media prep, speech processing, and annotation export."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    video_ids = resolve_video_ids(args)
    all_chunk_json_paths = []
    episode_json_paths = []
    failed_video_ids = []
    output_dir = Path(args.output_dir)
    manifest = new_manifest(video_ids)
    write_manifest(manifest, output_dir)

    for video_id in video_ids:
        try:
            chunk_json_paths, episode_json_path = process_episode(
                video_id,
                args,
                manifest,
            )
        except Exception as exc:
            LOGGER.exception("Failed episode %s: %s", video_id, exc)
            mark_episode(manifest, video_id, "failed")
            failed_video_ids.append(video_id)
            write_manifest(manifest, output_dir)
            if args.stop_on_error:
                raise
            continue
        all_chunk_json_paths.extend(chunk_json_paths)
        episode_json_paths.append(episode_json_path)
        mark_episode(manifest, video_id, "completed")
        write_manifest(manifest, output_dir)

    build_label_studio_tasks(
        all_chunk_json_paths,
        output_dir / "label_studio_tasks.json",
        video_url_prefix=args.video_url_prefix,
    )
    for video_id in video_ids:
        if video_id not in failed_video_ids:
            mark_stage(manifest, video_id, "label_studio_export", "completed")

    build_manual_review_payload(
        episode_json_paths,
        output_dir / "manual_review_queue.json",
        sample_size=args.evidence_sample_size,
        seed=args.evidence_sample_seed,
    )
    for video_id in video_ids:
        if video_id not in failed_video_ids:
            mark_stage(manifest, video_id, "roster_extraction", "pending_manual")
            mark_stage(manifest, video_id, "evidence_verification", "pending_manual")
            mark_stage(manifest, video_id, "speaker_name_linking", "pending_manual")

    write_manifest(manifest, output_dir)
    LOGGER.info("Done. Label Studio tasks written to %s", args.output_dir)
    LOGGER.info("Episodes requested: %s", len(video_ids))
    LOGGER.info("Episodes succeeded: %s", len(video_ids) - len(failed_video_ids))
    LOGGER.info("Episodes failed: %s", len(failed_video_ids))
    if failed_video_ids:
        LOGGER.info("Failed IDs: %s", ", ".join(failed_video_ids))


def process_episode(
    video_id: str,
    args: argparse.Namespace,
    manifest: Dict,
) -> Tuple[List[Path], Path]:
    """Process one episode and return its chunk JSON paths."""
    LOGGER.info("Processing episode %s", video_id)
    paths = prepare_media(video_id)
    mark_stage(manifest, video_id, "download_video", "completed")
    mark_stage(manifest, video_id, "extract_audio", "completed")
    mark_stage(manifest, video_id, "create_proxy", "completed")

    episode_dir = Path(args.output_dir) / video_id
    whisperx_path = episode_dir / "whisperx.json"
    osd_path = episode_dir / "pyannote_osd.json"
    overlaps_path = episode_dir / "overlap_candidates.json"
    final_json_path = episode_dir / "episode.json"

    whisperx_result = run_whisperx(
        Path(paths["audio"]),
        whisperx_path,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        language="en",
    )
    mark_stage(manifest, video_id, "whisperx", "completed")
    if args.skip_osd:
        LOGGER.warning("Skipping Pyannote OSD; using diarization overlaps only.")
        osd_intervals = []
        save_json(osd_intervals, osd_path)
        mark_stage(manifest, video_id, "pyannote_osd", "skipped")
    else:
        osd_intervals = run_pyannote_osd(Path(paths["audio"]), osd_path)
        mark_stage(manifest, video_id, "pyannote_osd", "completed")
    diarization_overlaps = diarization_intersections(
        whisperx_result.get("diarization", [])
    )
    overlap_candidates = merge_overlap_candidates(
        osd_intervals,
        diarization_overlaps,
    )
    save_json(overlap_candidates, overlaps_path)
    mark_stage(manifest, video_id, "merge_overlap_candidates", "completed")

    build_episode_json(
        episode_id=video_id,
        metadata_path=Path(paths["metadata"]),
        whisperx_path=whisperx_path,
        osd_path=osd_path,
        overlap_candidates=overlap_candidates,
        proxy_path=Path(paths["proxy"]),
        audio_path=Path(paths["audio"]),
        output_path=final_json_path,
    )
    mark_stage(manifest, video_id, "build_episode_json", "completed")

    chunk_paths = chunk_proxy_video(
        Path(paths["proxy"]),
        episode_dir / "chunks" / "videos",
        chunk_seconds=args.chunk_seconds,
    )
    mark_stage(manifest, video_id, "chunk_proxy_video", "completed")
    chunk_json_paths = slice_episode_json_for_chunks(
        final_json_path,
        chunk_paths,
        episode_dir / "chunks" / "json",
        chunk_seconds=args.chunk_seconds,
    )
    mark_stage(manifest, video_id, "slice_chunk_json", "completed")
    return chunk_json_paths, final_json_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-url", help="YouTube channel/videos URL.")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start-date", help="YYYY-MM-DD. Defaults to last month.")
    parser.add_argument("--end-date", help="YYYY-MM-DD. Defaults to last month.")
    parser.add_argument("--output-dir", default="data/english_news")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-seconds", type=int, default=900)
    parser.add_argument("--video-url-prefix", default="")
    parser.add_argument("--evidence-sample-size", type=int, default=10)
    parser.add_argument("--evidence-sample-seed", type=int, default=42)
    parser.add_argument(
        "--skip-osd",
        action="store_true",
        help="Debug option: skip Pyannote OSD and use diarization overlaps only.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when one episode fails.",
    )
    return parser.parse_args()


def resolve_video_ids(args: argparse.Namespace) -> List[str]:
    """Resolve explicit video IDs or discover episodes from a channel URL."""
    if args.video_id:
        return args.video_id[: args.limit]

    if not args.channel_url:
        raise SystemExit("Provide --channel-url or one or more --video-id values.")

    start_date, end_date = parse_date_range(args.start_date, args.end_date)
    candidates = discover_channel_episodes(
        args.channel_url,
        start_date=start_date,
        end_date=end_date,
        limit=args.limit,
    )
    return [candidate.video_id for candidate in candidates]


def parse_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
):
    """Parse CLI date range or return the previous calendar month."""
    if start_date and end_date:
        return (
            datetime.strptime(start_date, "%Y-%m-%d").date(),
            datetime.strptime(end_date, "%Y-%m-%d").date(),
        )
    return last_calendar_month()


if __name__ == "__main__":
    main()
