"""Step 11: chunk one episode's proxy video + JSON into 15-minute pieces."""

import argparse
import logging
from pathlib import Path

from v2.chunk.chunking import build_chunk_payload, chunk_plan, chunk_video
from v2.io import load_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.chunk")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut the proxy into 15-minute chunks and slice episode JSON."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=15 * 60,
        help="Chunk length in seconds (default 900).",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    layout = build_layout(args.video_id, args.data_dir)
    if not layout.final_path.exists():
        LOGGER.error("No final.json yet — run `python -m v2.cli.finalize` first.")
        raise SystemExit(1)

    final = load_json(layout.final_path)
    duration_s = (final.get("episode") or {}).get("duration_s") or 0.0
    if not duration_s:
        LOGGER.warning("No duration in metadata; chunking by final.json end time.")
        last_end = max(
            [s.get("end", 0) for s in final.get("segments", [])]
            + [s.get("start", 0) for s in final.get("segments", [])],
            default=0.0,
        )
        duration_s = last_end or args.chunk_seconds

    plan = chunk_video(
        layout.proxy_path,
        layout.chunks_dir,
        duration_s,
        chunk_seconds=args.chunk_seconds,
    )

    segments     = final.get("segments", [])
    candidates   = final.get("overlap_candidates", [])
    interruptions = final.get("interruptions", [])

    payloads = []
    for chunk in plan:
        payload = build_chunk_payload(
            final,
            chunk,
            segments,
            candidates,
            interruptions,
            video_path=chunk["video_path"],
        )
        chunk_path = layout.chunks_dir / f"chunk_{chunk['index']:03d}.json"
        from v2.io import save_json
        save_json(payload, chunk_path)
        payloads.append(payload)
        LOGGER.info("chunk %03d: %6.1f–%6.1f s  (%d segs, %d candidates)",
                    chunk["index"], chunk["start"], chunk["end"],
                    len(payload["segments"]), len(payload["overlap_candidates"]))

    LOGGER.info("Chunked %s into %d pieces under %s",
                args.video_id, len(plan), layout.chunks_dir)


if __name__ == "__main__":
    main()