"""Step 12: export a video's chunks to a Label Studio import file."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from v2.io import load_json
from v2.labelstudio.export import build_label_studio_tasks, write_tasks_jsonl
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.labelstudio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSONL import file for Label Studio (one task per chunk)."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--base-url",
        default=None,
        help="If set, chunk videos are referenced as <base-url>/chunk_000.mp4 "
        "(e.g. the Label Studio media url). Default: absolute file paths.",
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

    chunk_payloads = []
    if not layout.chunks_dir.exists():
        LOGGER.error("No chunks yet — run `python -m v2.cli.chunk` first.")
        raise SystemExit(1)
    for chunk_json in sorted(layout.chunks_dir.glob("chunk_*.json")):
        chunk_payloads.append(load_json(chunk_json))

    tasks = build_label_studio_tasks(chunk_payloads, args.video_id, base_url=args.base_url)
    write_tasks_jsonl(tasks, layout.labelstudio_tasks_path)
    LOGGER.info("Wrote %d tasks -> %s", len(tasks), layout.labelstudio_tasks_path)
    print(json.dumps(tasks[0], indent=2))


if __name__ == "__main__":
    main()