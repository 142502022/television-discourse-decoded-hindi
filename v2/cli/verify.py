"""Step 6: reconcile OSD and diarization into overlapping candidate intervals."""

import argparse
import logging
from pathlib import Path

from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.osd.verify import build_verification

LOGGER = logging.getLogger("v2.cli.verify")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile overlap alarms from OSD and diarization "
        "into three-case candidate intervals (AGREE / OSD_ONLY / "
        "DIARIZATION_ONLY)."
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    layout = build_layout(args.video_id, args.data_dir)

    overlap_payload = load_json(layout.overlap_path)
    diarization = load_json(layout.diarization_path)

    verification = build_verification(
        overlap_payload["regions"],
        diarization["turns"],
    )
    summary = verification["summary"]
    LOGGER.info(
        "Candidates: %d total | AGREE %d | OSD_ONLY %d | DIARIZATION_ONLY %d",
        summary["num_candidates"],
        summary["num_agree"],
        summary["num_osd_only"],
        summary["num_diarization_only"],
    )
    save_json(verification, layout.candidates_path)
    LOGGER.info("Overlap candidates saved: %s", layout.candidates_path)


if __name__ == "__main__":
    main()