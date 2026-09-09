"""Step 10: assemble a single research-grade JSON per episode.

Pure assembly — reads every artifact produced so far and writes a unified
``final.json`` that is ready for annotation import, statistics, and
downstream analysis (Label Studio, paper tables).
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.finalize")


def _safe(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return None
    return None


def build_final(layout: Any) -> Dict[str, Any]:
    """Aggregate every stage into one JSON."""
    metadata   = _safe(layout.metadata_path) or {}
    transcribe = _safe(layout.asr_path)
    diariz     = _safe(layout.diarization_path)
    diarized   = _safe(layout.diarized_transcript_path)
    osd_data   = _safe(layout.analysis_path)
    cand       = _safe(layout.candidates_path)
    linked     = _safe(layout.linked_path)
    roles      = _safe(layout.roles_path)

    # ── episode metadata ──────────────────────────────────────────────────
    episode = {
        "video_id":      metadata.get("id", layout.video_id),
        "title":         metadata.get("title"),
        "channel":       metadata.get("channel"),
        "channel_id":    metadata.get("channel_id"),
        "upload_date":   metadata.get("upload_date"),
        "duration_s":    metadata.get("duration"),
        "url":           metadata.get("webpage_url"),
        "thumbnail":     metadata.get("thumbnail"),
    }

    # ── pipeline provenance ──────────────────────────────────────────────
    pipeline = {}
    if transcribe:
        pipeline["asr"] = {
            "model":     transcribe.get("model"),
            "language":  transcribe.get("language"),
            "segments":  len(transcribe.get("segments", [])),
            "words":     sum(len(s.get("words", [])) for s in transcribe.get("segments", [])),
            "device":    transcribe.get("device"),
        }
    if diariz:
        pipeline["diarization"] = {
            "model":         diariz.get("model"),
            "device":        diariz.get("device"),
            "num_speakers":  diariz.get("num_speakers"),
            "turns":         len(diariz.get("turns", [])),
        }
    if osd_data:
        stats = osd_data.get("stats", {})
        pipeline["osd"] = {
            "model":           osd_data.get("detector", {}).get("model"),
            "device":          osd_data.get("detector", {}).get("device"),
            "overlap_regions": stats.get("num_regions"),
            "overlap_seconds": stats.get("overlap_seconds"),
            "num_interruptions": stats.get("num_interruptions"),
        }
    if cand:
        summary = cand.get("summary", {})
        pipeline["verify"] = {
            "method":             cand.get("method"),
            "num_candidates":     summary.get("num_candidates"),
            "agree":              summary.get("num_agree"),
            "osd_only":           summary.get("num_osd_only"),
            "diarization_only":   summary.get("num_diarization_only"),
        }
    if linked:
        pipeline["roster"] = {
            "model":  linked.get("method"),
        }
    if roles:
        pipeline["identity"] = {
            "method": roles.get("method"),
            "anchor": roles.get("anchor"),
        }

    # ── participants ─────────────────────────────────────────────────────
    participants = (linked or {}).get("participants", [])
    # Supplement heuristic-only entries with roster data if available
    roster = _safe(layout.roster_path)
    roster_by_label = {}
    if roster:
        for p in roster.get("participants", []):
            roster_by_label[p.get("speaker_label")] = p
    for entry in participants:
        r = roster_by_label.get(entry.get("speaker_label"), {})
        if r.get("name") and entry.get("name") == "Unknown":
            entry["name"] = r["name"]
        for key in ("affiliation", "gender_as_addressed", "evidence_quote", "evidence_timestamps_s"):
            if r.get(key) and not entry.get(key):
                entry[key] = r[key]

    # ── diarized segments & turns ────────────────────────────────────────
    segments    = (diarized or {}).get("segments", [])
    turns       = (diariz or {}).get("turns", [])
    candidates  = (cand or {}).get("candidates", [])
    interruptions = (osd_data or {}).get("interruptions", [])
    osd_stats   = (osd_data or {}).get("stats", {})

    return {
        "schema_version":      "1.0",
        "episode":             episode,
        "pipeline":            pipeline,
        "participants":        participants,
        "speaker_turns":       turns,
        "segments":            segments,
        "overlap_candidates":  candidates,
        "interruptions":       interruptions,
        "osd_stats":           osd_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble every stage into a single final.json per episode."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--data-dir", type=Path,
                        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    layout = build_layout(args.video_id, args.data_dir)
    final  = build_final(layout)
    save_json(final, layout.final_path)
    LOGGER.info(
        "final.json saved: %s  (%d segments, %d participants, %d candidates)",
        layout.final_path,
        len(final["segments"]),
        len(final["participants"]),
        len(final["overlap_candidates"]),
    )


if __name__ == "__main__":
    main()
