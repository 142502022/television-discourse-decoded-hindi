"""Step 11: chunk the proxy video (and episode JSON) into 15-minute pieces.

Every chunk is a self-contained unit: a playable mp4 cut plus a sliced
``chunk.json`` carrying the segments, overlap candidates, and interruptions
that fall inside that window. This is the unit Label Studio receives as one
task.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

CHUNK_SECONDS = 15 * 60  # 15 minutes per chunk


def chunk_plan(duration_s: float, chunk_seconds: int = CHUNK_SECONDS) -> List[Dict[str, Any]]:
    """Return chunk boundaries covering ``duration_s`` seconds."""
    n = max(1, -(-int(duration_s) // chunk_seconds))
    return [
        {
            "index": i,
            "start": i * chunk_seconds,
            "end": min((i + 1) * chunk_seconds, int(duration_s)),
        }
        for i in range(n)
    ]


def _overlaps(interval: Dict[str, Any], start: float, end: float) -> bool:
    return interval.get("end", 0) > start and interval.get("start", 0) < end


def slice_segments(segments: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    """Copy transcript segments that intersect the window, clipped to it."""
    sliced = []
    for seg in segments:
        if _overlaps(seg, start, end):
            clipped = dict(seg)
            clipped["start"] = max(seg.get("start", 0), start)
            clipped["end"] = min(seg.get("end", 0), end)
            sliced.append(clipped)
    return sliced


def slice_events(
    events: List[Dict[str, Any]],
    start: float,
    end: float,
) -> List[Dict[str, Any]]:
    """Copy overlap/interruption events that intersect the window, clipped."""
    return slice_segments(events, start, end)


def cut_chunk(
    proxy_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> Path:
    """Cut a playable mp4 chunk from the proxy using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(proxy_path),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def build_chunk_payload(
    episode: Dict[str, Any],
    chunk: Dict[str, Any],
    segments: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    interruptions: List[Dict[str, Any]],
    video_path: str,
) -> Dict[str, Any]:
    """Assemble one chunk's annotation payload."""
    start, end = chunk["start"], chunk["end"]
    return {
        "schema_version": "1.0",
        "episode_id": episode.get("video_id"),
        "chunk_index": chunk["index"],
        "start_s": start,
        "end_s": end,
        "duration_s": round(end - start, 3),
        "video": video_path,
        "participants": episode.get("participants", []),
        "segments": slice_segments(segments, start, end),
        "overlap_candidates": slice_events(candidates, start, end),
        "interruptions": slice_events(interruptions, start, end),
        "osd_stats": episode.get("osd_stats", {}),
    }


def chunk_video(
    proxy_path: Path,
    output_dir: Path,
    duration_s: float,
    chunk_seconds: int = CHUNK_SECONDS,
) -> List[Dict[str, Any]]:
    """Cut the proxy into chunk_seconds-long mp4 pieces, returning plans."""
    plan = chunk_plan(duration_s, chunk_seconds)
    for chunk in plan:
        output = output_dir / f"chunk_{chunk['index']:03d}.mp4"
        if not output.exists():
            cut_chunk(proxy_path, output, chunk["start"], chunk["end"])
        chunk["video_path"] = str(output)
    return plan