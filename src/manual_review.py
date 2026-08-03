import random
from pathlib import Path
from typing import Any, Dict, List

from src.speech.io import load_json, save_json


def build_manual_review_payload(
    episode_json_paths: List[Path],
    output_path: Path,
    sample_size: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a manual-review artifact for roster evidence and speaker linking."""
    evidence_candidates = []
    speaker_clusters = {}

    for episode_json_path in episode_json_paths:
        episode = load_json(episode_json_path)
        episode_id = episode["episode_id"]

        for segment in episode.get("segments", []):
            start = segment.get("start")
            text = str(segment.get("text") or "").strip()
            speaker = segment.get("speaker")
            if start is not None and text:
                evidence_candidates.append(
                    {
                        "episode_id": episode_id,
                        "timestamp": float(start),
                        "speaker": speaker,
                        "quote": text,
                        "proxy_video": episode.get("proxy_video"),
                        "review_status": "pending",
                        "review_notes": "",
                    }
                )
            if speaker:
                speaker_clusters.setdefault(
                    f"{episode_id}:{speaker}",
                    {
                        "episode_id": episode_id,
                        "anonymous_speaker": speaker,
                        "participant_name": None,
                        "role": None,
                        "affiliation": None,
                        "gender_as_addressed": None,
                        "confidence": "unverified",
                        "evidence": [],
                    },
                )

    sampled_evidence = _sample_evidence(evidence_candidates, sample_size, seed)
    payload = {
        "participants": [],
        "speaker_clusters_to_link": list(speaker_clusters.values()),
        "evidence_sample": sampled_evidence,
        "evidence_audit": {
            "sample_size": len(sampled_evidence),
            "checked": 0,
            "errors": 0,
            "accuracy": None,
            "status": "pending_manual_review",
        },
        "notes": [
            "Fill participants after LLM roster extraction.",
            "Manually verify evidence_sample timestamps in proxy videos.",
            "Link anonymous speaker clusters only after evidence is verified.",
        ],
    }
    save_json(payload, output_path)
    return payload


def _sample_evidence(
    evidence_candidates: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(evidence_candidates) <= sample_size:
        return evidence_candidates
    rng = random.Random(seed)
    return rng.sample(evidence_candidates, sample_size)
