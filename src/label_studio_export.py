import json
from pathlib import Path
from typing import Dict, List


def build_label_studio_tasks(
    chunk_json_paths: List[Path],
    output_path: Path,
    video_url_prefix: str = "",
) -> List[Dict]:
    """Create one Label Studio import task per chunk JSON file."""
    tasks = []

    for chunk_json_path in chunk_json_paths:
        with chunk_json_path.open("r", encoding="utf-8") as file_obj:
            chunk = json.load(file_obj)

        video_path = Path(chunk["proxy_video"])
        video_value = (
            f"{video_url_prefix.rstrip('/')}/{video_path.name}"
            if video_url_prefix
            else str(video_path)
        )
        tasks.append(
            {
                "data": {
                    "video": video_value,
                    "episode_id": chunk["episode_id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_start": chunk["chunk_start"],
                    "chunk_end": chunk["chunk_end"],
                    "overlap_candidates": chunk.get("overlap_candidates", []),
                    "segments": chunk.get("segments", []),
                },
                "meta": {
                    "chunk_json": str(chunk_json_path),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(tasks, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")

    return tasks
