"""Phase 2: infer the host/moderator of one episode.

Reads the assembled ``final.json`` for a video and writes an interpretable
host-identification artifact. The optional ``--reference-label`` is used only
to evaluate the prediction; it never changes the ranking.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from v2.identity.host import build_result
from v2.io import load_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.host")

DEFAULT_OUTPUT = Path("data/v2/analysis/host_identification.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer the host/moderator of one episode from final.json."
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--data-dir", type=Path,
        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument(
        "--reference-label", default=None,
        help="Manually supplied reference speaker label for evaluation only, "
             "e.g. SPEAKER_24. Never used to rank candidates.",
    )
    parser.add_argument("--trim-start", type=float, default=0.0,
                        help="Ad/intro trim boundary in seconds (metadata only).")
    parser.add_argument("--top-n", type=int, default=8,
                        help="How many ranked candidates to report (default: 8).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    layout = build_layout(args.video_id, args.data_dir)
    final_json = load_json(layout.final_path)

    result = build_result(
        final_json,
        video_id=args.video_id,
        reference_label=args.reference_label,
        trim_start_s=args.trim_start,
        top_n=args.top_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    prediction = result["prediction"]
    comparison = result["reference_comparison"]
    print(f"Predicted host: {prediction['speaker_label']} "
          f"(role={prediction['inferred_role']}, confidence={prediction['confidence_band']})")
    if args.reference_label:
        print(f"Reference: {comparison['reference_label']} | match={comparison['match']} "
              f"| reference_rank={comparison['reference_rank']}")
    for candidate in result["candidates"][:5]:
        print(f"  #{candidate['rank']} {candidate['speaker_label']:>12} "
              f"score={candidate['heuristic_score']}")
    print(f"Artifact saved to {args.output}")


if __name__ == "__main__":
    main()
