from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.speech.io import save_json


STAGE_ORDER = [
    "download_video",
    "extract_audio",
    "create_proxy",
    "whisperx",
    "pyannote_osd",
    "merge_overlap_candidates",
    "build_episode_json",
    "chunk_proxy_video",
    "slice_chunk_json",
    "label_studio_export",
    "roster_extraction",
    "evidence_verification",
    "speaker_name_linking",
]


def new_manifest(video_ids: List[str]) -> Dict[str, Any]:
    """Create a run manifest for tracking completed and pending stages."""
    return {
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "episodes_requested": video_ids,
        "episodes": {
            video_id: {
                "status": "pending",
                "stages": {
                    stage: {
                        "status": "pending",
                        "error": None,
                    }
                    for stage in STAGE_ORDER
                },
            }
            for video_id in video_ids
        },
    }


def mark_stage(
    manifest: Dict[str, Any],
    video_id: str,
    stage: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update one stage status in the run manifest."""
    manifest["episodes"][video_id]["stages"][stage] = {
        "status": status,
        "error": error,
    }
    manifest["updated_at"] = _utc_now()


def mark_episode(
    manifest: Dict[str, Any],
    video_id: str,
    status: str,
) -> None:
    """Update one episode's overall status."""
    manifest["episodes"][video_id]["status"] = status
    manifest["updated_at"] = _utc_now()


def write_manifest(manifest: Dict[str, Any], output_dir: Path) -> Path:
    """Write the pipeline status manifest to disk."""
    output_path = output_dir / "pipeline_status.json"
    save_json(manifest, output_path)
    return output_path


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
