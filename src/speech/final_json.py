from pathlib import Path
from typing import Any, Dict, List

from .io import load_json, save_json
from .roster import empty_roster_payload, speaker_name_links


def build_episode_json(
    episode_id: str,
    metadata_path: Path,
    whisperx_path: Path,
    osd_path: Path,
    overlap_candidates: List[Dict[str, Any]],
    proxy_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build the final per-episode JSON for downstream annotation."""
    metadata = load_json(metadata_path)
    whisperx = load_json(whisperx_path)
    osd_intervals = load_json(osd_path)
    roster_payload = empty_roster_payload()

    episode_json = {
        "episode_id": episode_id,
        "source_url": metadata.get("webpage_url"),
        "title": metadata.get("title"),
        "upload_date": metadata.get("upload_date"),
        "duration": metadata.get("duration"),
        "proxy_video": str(proxy_path),
        "audio": str(audio_path),
        "segments": whisperx.get("segments", []),
        "diarization": whisperx.get("diarization", []),
        "pyannote_osd_intervals": osd_intervals,
        "overlap_candidates": overlap_candidates,
        "participants": roster_payload["participants"],
        "speaker_links": speaker_name_links(),
        "evidence_audit": roster_payload["evidence_audit"],
    }
    save_json(episode_json, output_path)
    return episode_json
