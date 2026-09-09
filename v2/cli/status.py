"""Print a per-video artifact presence table for the Makefile status target."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KEYS = [
    ("source/original.mp4", "M1"),
    ("audio/audio.wav", "M1"),
    ("proxy/proxy_480p.mp4", "M1"),
    ("asr/transcription.json", "M2"),
    ("asr/diarized_transcript.json", "M3"),
    ("osd/analysis.json", "M4"),
    ("osd/candidates.json", "S6"),
    ("identity/roles.json", "S5"),
    ("roster/roster.json", "S7"),
]

IDS = json.loads((ROOT / "ids.json").read_text())


def main() -> None:
    header = "\t".join(["VIDEO_ID"] + [label for _, label in KEYS])
    print(header)
    for vid in IDS:
        row = [vid]
        for relative, _ in KEYS:
            row.append("Y" if (ROOT / "data/v2/videos" / vid / relative).exists() else "-")
        print("\t".join(row))


if __name__ == "__main__":
    main()