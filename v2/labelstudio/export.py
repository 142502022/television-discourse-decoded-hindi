"""Step 12: build Label Studio import files.

One task per chunk. Label Studio consumes a JSONL file where each line is a
task:

    {"data": {"video": "<url or path>", "episode_id": "...", "chunk_index": 0}, "annotations": []}

The video is referenced by URL when ``base_url`` is given (media served from
Label Studio's storage) or by absolute path when not.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_task(
    chunk: Dict[str, Any],
    episode_id: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn one chunk payload into a Label Studio task."""
    video_source = chunk["video"]
    if base_url and not video_source.startswith(("http://", "https://")):
        video_source = f"{base_url.rstrip('/')}/{Path(video_source).name}"
    return {
        "data": {
            "video": video_source,
            "episode_id": episode_id,
            "chunk_index": chunk.get("chunk_index"),
            "start_s": chunk.get("start_s"),
            "end_s": chunk.get("end_s"),
            "overlap_candidates": chunk.get("overlap_candidates", []),
            "segments": chunk.get("segments", []),
        },
        "annotations": [],
        "predictions": [],
    }


def build_label_studio_tasks(
    chunk_payloads: List[Dict[str, Any]],
    episode_id: str,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        build_task(payload, episode_id, base_url=base_url)
        for payload in sorted(chunk_payloads, key=lambda c: c["chunk_index"])
    ]


def write_tasks_jsonl(
    tasks: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return output_path