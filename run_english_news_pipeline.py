#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.label_studio_export import build_label_studio_tasks
from src.media.chunking import chunk_proxy_video, slice_episode_json_for_chunks
from src.media.discovery import discover_channel_episodes, last_calendar_month
from src.media.pipeline import prepare_media
from src.speech.final_json import build_episode_json
from src.speech.io import load_json, save_json
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

    for video_id in video_ids:
        LOGGER.info("Processing episode %s", video_id)
        paths = prepare_media(video_id)

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
        osd_intervals = run_pyannote_osd(Path(paths["audio"]), osd_path)
        diarization_overlaps = diarization_intersections(
            whisperx_result.get("diarization", [])
        )
        overlap_candidates = merge_overlap_candidates(
            osd_intervals,
            diarization_overlaps,
        )
        save_json(overlap_candidates, overlaps_path)

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

        chunk_paths = chunk_proxy_video(
            Path(paths["proxy"]),
            episode_dir / "chunks" / "videos",
            chunk_seconds=args.chunk_seconds,
        )
        chunk_json_paths = slice_episode_json_for_chunks(
            final_json_path,
            chunk_paths,
            episode_dir / "chunks" / "json",
            chunk_seconds=args.chunk_seconds,
        )
        all_chunk_json_paths.extend(chunk_json_paths)

    build_label_studio_tasks(
        all_chunk_json_paths,
        Path(args.output_dir) / "label_studio_tasks.json",
        video_url_prefix=args.video_url_prefix,
    )
    LOGGER.info("Done. Label Studio tasks written to %s", args.output_dir)


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
