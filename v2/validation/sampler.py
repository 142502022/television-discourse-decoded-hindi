"""Step 8: validation sampler — pick random evidence timestamps across the
corpus and build a verification harness.

Partially manual: the sampler generates candidate timestamps with quotes and
expected speakers; the annotator jumps into the proxy video, confirms or
rejects, and then ``compute_stats`` produces the final accuracy numbers.
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from v2.io import load_json, save_json


def sample_timestamps(
    per_video: List[Dict[str, Any]],
    n: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pick ``n`` random evidence timestamps across the corpus.

    Each entry in ``per_video`` has ``video_id`` and ``segments`` (from the
    diarized transcript).  A segment is sampled uniformly across the corpus
    then within segments, weighted by duration (longer segments more likely to
    contain verifiable content).
    """
    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []
    total_duration = 0.0
    for v in per_video:
        for s in v.get("segments", []):
            dur = (s.get("end", 0) - s.get("start", 0))
            if dur > 0:
                total_duration += dur
                candidates.append({
                    "video_id":    v["video_id"],
                    "start":       s["start"],
                    "end":         s["end"],
                    "speaker":     s.get("speaker", "UNKNOWN"),
                    "text":        s.get("text", "").strip(),
                })

    if not candidates:
        return []

    weights = [(c["end"] - c["start"]) for c in candidates]
    chosen = []
    chosen_set: set = set()
    attempts = 0
    while len(chosen) < min(n, len(candidates)) and attempts < n * 20:
        idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        if idx not in chosen_set:
            chosen_set.add(idx)
            chosen.append(candidates[idx])
        attempts += 1

    chosen.sort(key=lambda c: (c["video_id"], c["start"]))
    return [
        {
            "video_id":         c["video_id"],
            "timestamp_s":      round(c["start"], 3),
            "speaker_label":    c["speaker"],
            "expected_text":    c["text"],
            "annulment":        None,  # annotator fills: pass / fail / unclear
            "comment":          None,
        }
        for c in chosen
    ]


def build_sample_artifact(
    samples: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    return {
        "method":     "random-evidence-sample",
        "model":      model,
        "n_samples":  len(samples),
        "samples":    samples,
    }


def create_clip(
    proxy_path: Path,
    timestamp_s: float,
    output_path: Path,
    clip_duration: float = 10.0,
    before: float = 2.0,
) -> None:
    """Cut a short clip around ``timestamp_s`` from the proxy video."""
    start = max(0.0, timestamp_s - before)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(proxy_path),
        "-t", str(clip_duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def create_all_clips(
    samples: List[Dict[str, Any]],
    data_root: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate ffmpeg clips for every sample that has a proxy video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, s in enumerate(samples):
        proxy = data_root / s["video_id"] / "proxy" / "proxy_480p.mp4"
        if not proxy.exists():
            continue
        clip = output_dir / f"sample_{i:02d}_{s['video_id']}_{s['speaker_label']}.mp4"
        try:
            create_clip(proxy, s["timestamp_s"], clip)
            paths.append(clip)
        except Exception:
            continue
    return paths


def compute_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy from annotator marks (pass / fail / unclear)."""
    pass_count = sum(1 for s in samples if s.get("annulment") == "pass")
    fail_count = sum(1 for s in samples if s.get("annulment") == "fail")
    unclear    = sum(1 for s in samples if s.get("annulment") == "unclear")
    unanswered = sum(1 for s in samples if s.get("annulment") is None)
    answered   = pass_count + fail_count + unclear
    total      = len(samples)
    return {
        "total":       total,
        "answered":    answered,
        "pass":        pass_count,
        "fail":        fail_count,
        "unclear":     unclear,
        "unanswered":  unanswered,
        "accuracy":    round(pass_count / answered, 4) if answered else None,
        "error_rate":  round(fail_count / answered, 4) if answered else None,
    }
