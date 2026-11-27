"""Identify one overall topic per video and analysis variant.

The command reads only the variant-local ``merge_final.json`` and the video's
curated metadata.  It deliberately does not read advertisement-detection
artifacts or alter any pipeline input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "v2" / "debate_only" / "videos"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "v2" / "debate_only" / "overall_topic"
VARIANTS = ("baseline", "threshold_0.5", "threshold_1.0", "threshold_2.0", "threshold_5.0")
DEFAULT_MODEL = "gemini-3.8-flash-lite"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

WINDOW_SYSTEM_INSTRUCTIONS = """You analyse one time window from a televised debate.
Use only supplied metadata and transcript. Transcript is primary; metadata is context only.
Return JSON only with main_issue (string or null), summary (string or null),
discussion_points (array of strings), entities (array of strings), evidence (array of
objects with start, end, speaker_id, text), and uncertainty (string or null).
Evidence text must be an exact excerpt from a supplied transcript segment. Do not invent
events, entities, arguments, or evidence. This is not subtopic detection."""

EPISODE_SYSTEM_INSTRUCTIONS = """Identify exactly one overall topic for a televised debate.
Use only the supplied metadata and window analyses. Do not blindly copy metadata title.
Return JSON only with topic (non-empty string), description (string),
key_discussion_points (array of strings), supporting_evidence (array of objects with
start, end, speaker_id, text), topic_alignment (object with status: aligned,
partially_aligned, misaligned, or uncertain; and reason). Supporting evidence text must
be an exact excerpt present in the supplied window evidence. Do not create a subtopic
timeline or make unsupported factual claims."""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def extract_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return valid transcript records, retaining only actual ``segments`` text."""
    extracted: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        start, end = _number(segment.get("start")), _number(segment.get("end"))
        text = segment.get("text")
        if start is None or end is None or end < start or not isinstance(text, str) or not text.strip():
            continue
        speaker = segment.get("speaker")
        extracted.append({"start": start, "end": end, "speaker_id": speaker if isinstance(speaker, str) else None,
                          "text": text.strip()})
    return sorted(extracted, key=lambda item: (item["start"], item["end"]))


def build_windows(segments: list[dict[str, Any]], duration_seconds: float = 600.0) -> list[dict[str, Any]]:
    """Assign each segment once by its start timestamp; never split ASR sentences."""
    if duration_seconds <= 0:
        raise ValueError("window duration must be positive")
    if not segments:
        return []
    final_end = max(item["end"] for item in segments)
    windows: list[dict[str, Any]] = []
    start = 0.0
    index = 1
    while start < final_end:
        end = min(start + duration_seconds, final_end)
        members = [item for item in segments if start <= item["start"] < end]
        if members:
            windows.append({"window_id": f"W{index:03d}", "start": start, "end": end, "segments": members})
            index += 1
        start += duration_seconds
    return windows


def curated_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # The V2 curated schema presently has no description/tags/hashtags.
    return {"title": metadata.get("title") if isinstance(metadata.get("title"), str) else None,
            "description": metadata.get("description") if isinstance(metadata.get("description"), str) else None,
            "tags": metadata.get("tags") if isinstance(metadata.get("tags"), list) else [],
            "hashtags": metadata.get("hashtags") if isinstance(metadata.get("hashtags"), list) else [],
            "channel": metadata.get("channel") if isinstance(metadata.get("channel"), str) else None,
            "source_file": str(metadata.get("webpage_url") or "") or None}



def call_gemini(
    api_key: str,
    model: str,
    instructions: str,
    payload: dict[str, Any],
    timeout: int = 90,
) -> dict[str, Any]:

    request = urllib.request.Request(
        API_ENDPOINT.format(model=model, key=api_key),
        data=json.dumps(
            {
                "systemInstruction": {
                    "parts": [{"text": instructions}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": json.dumps(
                                    payload, ensure_ascii=False
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_attempts = 5
    retryable_status_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_attempts):

        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response:
                raw = json.loads(
                    response.read().decode("utf-8")
                )

            candidates = raw.get("candidates", [])

            text = "\n".join(
                part["text"]
                for candidate in candidates
                for part in candidate.get("content", {}).get("parts", [])
                if isinstance(part.get("text"), str)
            )

            if not text.strip():
                raise RuntimeError(
                    f"Gemini returned no text. Response: {raw}"
                )

            result = json.loads(text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini response JSON was not an object"
                )

            return result

        except urllib.error.HTTPError as exc:

            error_body = exc.read().decode(
                "utf-8", errors="replace"
            )

            print(
                f"Gemini HTTP {exc.code}: {error_body}",
                file=sys.stderr,
            )

            if exc.code not in retryable_status_codes:
                raise RuntimeError(
                    f"Non-retryable Gemini HTTP {exc.code}: "
                    f"{error_body}"
                ) from exc

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Gemini request failed after {max_attempts} "
                    f"attempts. HTTP {exc.code}: {error_body}"
                ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Gemini request failed after {max_attempts} "
                    f"attempts: {exc}"
                ) from exc

            print(
                f"Gemini connection error: {exc}",
                file=sys.stderr,
            )

        delay = min(15 * (2 ** attempt), 120)

        print(
            f"Retrying Gemini in {delay} seconds "
            f"(attempt {attempt + 2}/{max_attempts})...",
            file=sys.stderr,
        )

        time.sleep(delay)

    raise RuntimeError("Gemini request failed unexpectedly")


def _evidence(value: Any, allowed: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_records = list(allowed)
    valid: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = item["text"].strip()
        match = next((source for source in allowed_records if text and text in source["text"]), None)
        if match:
            valid.append({"start": match["start"], "end": match["end"], "speaker_id": match["speaker_id"], "text": text})
    return valid


def validate_window_result(result: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    points = result.get("discussion_points", [])
    return {"window_id": window["window_id"], "start": window["start"], "end": window["end"],
            "main_issue": result.get("main_issue") if isinstance(result.get("main_issue"), str) else None,
            "summary": result.get("summary") if isinstance(result.get("summary"), str) else None,
            "discussion_points": [x for x in points if isinstance(x, str)][:20],
            "entities": [x for x in result.get("entities", []) if isinstance(x, str)][:30],
            "evidence": _evidence(result.get("evidence"), window["segments"]),
            "uncertainty": result.get("uncertainty") if isinstance(result.get("uncertainty"), str) else None}


def validate_overall_result(result: dict[str, Any], analyses: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = [evidence for analysis in analyses for evidence in analysis["evidence"]]
    alignment = result.get("topic_alignment") if isinstance(result.get("topic_alignment"), dict) else {}
    status = alignment.get("status") if alignment.get("status") in {"aligned", "partially_aligned", "misaligned", "uncertain"} else "uncertain"
    topic = result.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("LLM overall result must include a non-empty topic")
    return {"topic": topic.strip(), "description": result.get("description") if isinstance(result.get("description"), str) else None,
            "key_discussion_points": [x for x in result.get("key_discussion_points", []) if isinstance(x, str)][:30],
            "supporting_evidence": _evidence(result.get("supporting_evidence"), allowed),
            "topic_alignment": {"status": status, "reason": alignment.get("reason") if isinstance(alignment.get("reason"), str) else None}}


def detect_one(video_id: str, variant: str, source: Path, metadata_path: Path, output: Path, *, api_key: str, model: str,
               llm: Callable[[str, str, str, dict[str, Any]], dict[str, Any]] = call_gemini) -> tuple[int, dict[str, Any]]:
    transcript = load_json(source)
    metadata = curated_metadata(load_json(metadata_path))
    windows = build_windows(extract_segments(transcript))
    if not windows:
        raise ValueError("No non-empty transcript windows were found in segments")
    analyses = []
    for window in windows:
        payload = {"metadata": metadata, "window": {key: window[key] for key in ("window_id", "start", "end")}, "segments": window["segments"]}
        analyses.append(validate_window_result(llm(api_key, model, WINDOW_SYSTEM_INSTRUCTIONS, payload), window))
    overall_payload = {"metadata": metadata, "windows": analyses}
    overall = validate_overall_result(llm(api_key, model, EPISODE_SYSTEM_INSTRUCTIONS, overall_payload), analyses)
    result = {"schema_version": "overall-topic-1.0", "video_id": video_id, "variant": variant, "source_file": str(source),
              "metadata": metadata, "configuration": {"window_duration_minutes": 10, "method": "metadata_and_transcript_llm", "model": model,
              "advertisement_detection_integrated": False}, "window_analysis": analyses, "overall_topic": {key: overall[key] for key in ("topic", "description", "key_discussion_points", "supporting_evidence")}, "topic_alignment": overall["topic_alignment"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return len(windows), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--video-id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is required for overall-topic identification.", file=sys.stderr)
        return 2
    variants = (args.variant,) if args.variant else VARIANTS
    video_dirs = [args.data_dir / args.video_id] if args.video_id else sorted(path for path in args.data_dir.iterdir() if path.is_dir())
    totals = Counter(videos_discovered=len(video_dirs), variants_discovered=len(variants))
    for video_dir in video_dirs:
        metadata_path = video_dir / "metadata" / "metadata.json"
        for variant in variants:
            source = video_dir / variant / "merge_final.json"
            if not source.is_file():
                totals["missing_transcript_files"] += 1; print(f"[WARNING] missing transcript: {source}", file=sys.stderr); continue
            if not metadata_path.is_file():
                totals["missing_metadata_files"] += 1; print(f"[WARNING] missing metadata: {metadata_path}", file=sys.stderr); continue
            totals["jobs_attempted"] += 1
            try:
                count, _ = detect_one(video_dir.name, variant, source, metadata_path, args.output_dir / variant / video_dir.name / "overall_topic.json", api_key=api_key, model=args.model)
                totals["successful_jobs"] += 1; totals["windows_processed"] += count
                print(f"[OK] {video_dir.name} / {variant}")
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                totals["failed_jobs"] += 1; print(f"[ERROR] {video_dir.name} / {variant}: {exc}", file=sys.stderr)
    print("Overall topic identification summary")
    for key in ("videos_discovered", "variants_discovered", "jobs_attempted", "successful_jobs", "failed_jobs", "missing_metadata_files", "missing_transcript_files", "windows_processed"):
        print(f"  {key.replace('_', ' ')}: {totals[key]}")
    print(f"  output directory: {args.output_dir}")
    return 1 if totals["failed_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
