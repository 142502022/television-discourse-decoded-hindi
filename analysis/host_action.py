#!/usr/bin/env python3
"""Generate segment-level host-action annotations from final.json using Gemini.

Only final.json['segments'] is used as transcript input. The script does not
modify source JSON files or interruption-management outputs.

Configuration:
  Edit VIDEOS_ROOT below to point to the root directory containing video-ID folders.

Example:
  export GEMINI_API_KEY='...'
  python generate_host_actions.py --limit-chunks 1
  python generate_host_actions.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-3.5-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OUTPUT_FILENAME = "host_actions.json"

# EDIT THIS PATH IN THE SCRIPT; no directory path is required on the command line.
# Relative to the directory from which you run the script (run from repository root).
VIDEOS_ROOT = Path("data/v2/videos")

ACTION_LABELS = [
    "challenging_participant",
    "asking_for_clarification",
    "allocating_floor",
    "regaining_control",
    "redirecting_topic",
    "allowing_guest_to_continue",
    "summarizing",
    "other_or_unclear",
]

SYSTEM_INSTRUCTION = """
You are annotating the host's conversational actions in a TV debate transcript.
Your job is to classify ONLY the supplied target segments spoken by the identified
host. Other speakers' segments are context and must not receive labels.

Use the surrounding transcript to interpret each target segment. Do not infer an
action from speaker order, timing, or punctuation alone. Do not invent words or
context. Transcript text may contain ASR errors or translated text; interpret
cautiously and quote evidence exactly from the supplied segment text.

Choose exactly one primary label for each target segment from this list:
- challenging_participant: contests, presses, questions the basis of, or demands
  justification for a participant's claim. A neutral request for explanation
  alone is not enough; use asking_for_clarification unless context indicates
  challenge or pressure.
- asking_for_clarification: requests explanation, repetition, or clarification
  without clear evidence of contestation.
- allocating_floor: explicitly invites, selects, or directs a participant to speak.
- regaining_control: actively re-establishes moderation control, for example by
  stopping competing speakers or asserting the right to manage the discussion.
  Do not use this merely because the host speaks after an overlap.
- redirecting_topic: explicitly shifts the discussion to a different issue or
  brings speakers back to the topic.
- allowing_guest_to_continue: explicitly signals or clearly indicates that a
  guest may continue speaking. Do not infer this only because the guest speaks next.
- summarizing: recaps or synthesizes positions or key points already discussed.
- other_or_unclear: host speech is present but none of the specific labels is
  supported, or the segment is too short/ambiguous to classify reliably.

Return one JSON object with key "actions", an array containing exactly one object
for every target segment. Each object must contain:
  segment_index: integer copied from input
  action: one allowed label
  confidence: number from 0 to 1 (not a calibrated probability)
  rationale: concise explanation grounded in transcript context
  evidence_quote: exact substring from that target segment's text, or empty string
    if no useful quote can be extracted

Do not return markdown fences or any text outside the JSON object. Do not omit a
target segment. If unsure, use other_or_unclear and lower confidence.
""".strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def resolve_host_id(video_dir: Path, final_data: dict[str, Any]) -> tuple[str | None, str | None]:
    candidates = [
        video_dir / "host_identification" / "host_identification.json",
        video_dir / "host_identification.json",
        video_dir.parent / "host_identification" / "host_identification.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                data = load_json(path)
                host = data.get("predicted_host_speaker") or data.get("host_speaker_id")
                if host:
                    return str(host), str(path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] Could not read host file {path}: {exc}", file=sys.stderr)

    # A fallback is permitted only if the source final.json itself explicitly
    # carries a host speaker ID. We do not infer the host from speech volume.
    host = final_data.get("host_speaker_id")
    if host:
        return str(host), "final.json:host_speaker_id"
    return None, None


def normalize_segments(final_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = final_data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("final.json has no list-valued 'segments' section")

    normalized: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = seg.get("speaker")
        # Some schemas may only have word-level speakers. Derive a speaker only
        # when all non-empty word-level speaker values agree; otherwise leave null.
        if not speaker:
            words = seg.get("words")
            if isinstance(words, list):
                word_speakers = {
                    w.get("speaker") for w in words
                    if isinstance(w, dict) and w.get("speaker")
                }
                if len(word_speakers) == 1:
                    speaker = next(iter(word_speakers))
        normalized.append({
            "segment_index": idx,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "speaker": speaker,
            "text": text.strip(),
        })
    normalized.sort(key=lambda x: (
        float(x["start"]) if isinstance(x.get("start"), (int, float)) else float("inf"),
        x["segment_index"],
    ))
    return normalized


def make_chunks(segments: list[dict[str, Any]], chunk_size: int, context_segments: int) -> list[dict[str, Any]]:
    """Chunk by ordered segment count; context segments are sent but not classified twice."""
    chunks = []
    for start in range(0, len(segments), chunk_size):
        end = min(start + chunk_size, len(segments))
        context_start = max(0, start - context_segments)
        context_end = min(len(segments), end + context_segments)
        chunks.append({
            "chunk_index": len(chunks),
            "core_start": start,
            "core_end": end,
            "core": segments[start:end],
            "context": segments[context_start:context_end],
        })
    return chunks


def call_gemini(api_key: str, model: str, payload: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    url = f"{API_BASE}/{model}:generateContent"
    request_body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "actions": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "segment_index": {"type": "INTEGER"},
                                "action": {"type": "STRING", "enum": ACTION_LABELS},
                                "confidence": {"type": "NUMBER"},
                                "rationale": {"type": "STRING"},
                                "evidence_quote": {"type": "STRING"},
                            },
                            "required": ["segment_index", "action", "confidence", "rationale", "evidence_quote"],
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    }
    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {json.dumps(response_data)[:1200]}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            if not text_parts:
                raise RuntimeError(f"Gemini response had no text: {json.dumps(response_data)[:1200]}")
            result = json.loads("\n".join(text_parts))
            if not isinstance(result, dict) or not isinstance(result.get("actions"), list):
                raise ValueError("Gemini response JSON must contain an 'actions' array")
            return result
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"HTTP {exc.code}: {body[:2000]}"
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(message) from exc
            if attempt >= retries:
                raise RuntimeError(message) from exc
            print(f"[WARN] Gemini request failed ({message}); retrying...", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            if attempt >= retries:
                raise RuntimeError(str(exc)) from exc
            print(f"[WARN] Gemini request failed ({exc}); retrying in {2 ** attempt}s...", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise RuntimeError("Gemini request failed after retries")


def validate_actions(
    returned: dict[str, Any],
    target_segments: list[dict[str, Any]],
    host_id: str,
) -> list[dict[str, Any]]:
    targets_by_index = {s["segment_index"]: s for s in target_segments}
    valid: dict[int, dict[str, Any]] = {}
    for item in returned.get("actions", []):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("segment_index"))
        except (TypeError, ValueError):
            continue
        if idx not in targets_by_index or idx in valid:
            continue
        action = item.get("action")
        if action not in ACTION_LABELS:
            action = "other_or_unclear"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        seg_text = targets_by_index[idx]["text"]
        quote = item.get("evidence_quote", "")
        if not isinstance(quote, str) or (quote and quote not in seg_text):
            quote = ""
        rationale = item.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = ""
        valid[idx] = {
            "segment_index": idx,
            "start": targets_by_index[idx].get("start"),
            "end": targets_by_index[idx].get("end"),
            "speaker": host_id,
            "text": seg_text,
            "host_action": action,
            "confidence": confidence,
            "rationale": rationale,
            "evidence_quote": quote,
            "review_status": "needs_manual_review",
        }

    # Preserve an explicit record for missing model results instead of silently
    # dropping a host segment from the output.
    for idx, seg in targets_by_index.items():
        if idx not in valid:
            valid[idx] = {
                "segment_index": idx,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": host_id,
                "text": seg["text"],
                "host_action": "other_or_unclear",
                "confidence": 0.0,
                "rationale": "No valid classification returned for this target segment.",
                "evidence_quote": "",
                "review_status": "classification_missing",
            }
    return [valid[idx] for idx in sorted(valid)]


def process_video(video_dir: Path, args: argparse.Namespace, api_key: str) -> tuple[bool, str]:
    final_path = video_dir / "final.json"
    if not final_path.is_file():
        return False, f"SKIP: missing {final_path}"
    output_path = video_dir / OUTPUT_FILENAME
    if output_path.exists() and not args.overwrite:
        return False, f"SKIP: {output_path} already exists (use --overwrite to replace)"

    try:
        final_data = load_json(final_path)
        if not isinstance(final_data, dict):
            return False, f"ERROR: {final_path} root must be a JSON object"
        host_id, host_source = resolve_host_id(video_dir, final_data)
        if not host_id:
            return False, f"ERROR: host ID not found; expected host_identification JSON under {video_dir}"
        segments = normalize_segments(final_data)
        if not segments:
            return False, f"ERROR: no usable transcript segments in {final_path}"
        host_segments = [s for s in segments if s.get("speaker") == host_id]
        if not host_segments:
            return False, f"ERROR: no segments in final.json['segments'] match host {host_id}"

        chunks = make_chunks(segments, args.chunk_size, args.context_segments)
        if args.limit_chunks is not None:
            chunks = chunks[:args.limit_chunks]
        all_actions: list[dict[str, Any]] = []
        completed_chunks: list[dict[str, Any]] = []
        failed_chunks: list[dict[str, Any]] = []
        done_target_ids: set[int] = set()

        print(f"[{video_dir.name}] host={host_id} ({host_source}); segments={len(segments)}; host_segments={len(host_segments)}; chunks={len(chunks)}")
        for chunk in chunks:
            core = chunk["core"]
            targets = [s for s in core if s.get("speaker") == host_id]
            if not targets:
                completed_chunks.append({"chunk_index": chunk["chunk_index"], "status": "no_host_segments", "target_segment_count": 0})
                continue
            # Context includes neighboring segments, but only core host segments are targets.
            payload = {
                "task": "Classify the host's action for each target segment using surrounding context.",
                "video_id": final_data.get("episode", {}).get("video_id", video_dir.name),
                "video_title": final_data.get("episode", {}).get("title"),
                "host_speaker_id": host_id,
                "allowed_action_labels": ACTION_LABELS,
                "context_segments": chunk["context"],
                "target_host_segments": targets,
            }
            try:
                returned = call_gemini(api_key, args.model, payload, args.timeout, args.retries)
                validated = validate_actions(returned, targets, host_id)
                all_actions.extend(validated)
                done_target_ids.update(a["segment_index"] for a in validated)
                completed_chunks.append({
                    "chunk_index": chunk["chunk_index"],
                    "status": "completed",
                    "target_segment_count": len(targets),
                    "returned_action_count": len(validated),
                })
                print(f"  chunk {chunk['chunk_index'] + 1}/{len(chunks)}: classified {len(validated)} host segments")
            except Exception as exc:
                failed_chunks.append({"chunk_index": chunk["chunk_index"], "error": str(exc), "target_segment_count": len(targets)})
                print(f"  [ERROR] chunk {chunk['chunk_index'] + 1}/{len(chunks)} failed: {exc}", file=sys.stderr)
            # Save progress after each chunk so completed work is not lost.
            progress_payload = build_output(final_data, host_id, host_source, all_actions, completed_chunks, failed_chunks, args.model, len(segments), len(host_segments), complete=False)
            save_json_atomic(output_path, progress_payload)

        # Add explicit missing records for host segments not processed due to chunk limit or failures.
        action_by_idx = {a["segment_index"]: a for a in all_actions}
        processed_chunk_indexes = {c["chunk_index"] for c in completed_chunks}
        failed_indexes = {f["chunk_index"] for f in failed_chunks}
        for chunk in make_chunks(segments, args.chunk_size, args.context_segments):
            if chunk["chunk_index"] in processed_chunk_indexes:
                continue
            targets = [s for s in chunk["core"] if s.get("speaker") == host_id]
            status = "not_processed_limit" if args.limit_chunks is not None and chunk["chunk_index"] >= args.limit_chunks else "chunk_failed"
            for seg in targets:
                if seg["segment_index"] not in action_by_idx:
                    action_by_idx[seg["segment_index"]] = {
                        "segment_index": seg["segment_index"], "start": seg.get("start"), "end": seg.get("end"),
                        "speaker": host_id, "text": seg["text"], "host_action": "other_or_unclear", "confidence": 0.0,
                        "rationale": "Not classified; see chunk_status for the reason.", "evidence_quote": "",
                        "review_status": status,
                    }
        final_actions = [action_by_idx[k] for k in sorted(action_by_idx)]
        complete = not failed_chunks and args.limit_chunks is None
        output = build_output(final_data, host_id, host_source, final_actions, completed_chunks, failed_chunks, args.model, len(segments), len(host_segments), complete=complete)
        save_json_atomic(output_path, output)
        return True, f"WROTE: {output_path} ({len(final_actions)} host segments; complete={complete}; failed_chunks={len(failed_chunks)})"
    except Exception as exc:
        return False, f"ERROR processing {video_dir}: {exc}"


def build_output(
    final_data: dict[str, Any], host_id: str, host_source: str | None, actions: list[dict[str, Any]],
    completed_chunks: list[dict[str, Any]], failed_chunks: list[dict[str, Any]], model: str,
    segment_count: int, host_segment_count: int, complete: bool,
) -> dict[str, Any]:
    episode = final_data.get("episode", {})
    return {
        "schema_version": "1.0",
        "video_id": episode.get("video_id"),
        "title": episode.get("title"),
        "source_file": "final.json",
        "transcript_source_section": "segments",
        "host_speaker_id": host_id,
        "host_identification_source": host_source,
        "model": model,
        "method": "Gemini-assisted segment-level host-action classification; manually review before research claims",
        "complete": complete,
        "summary": {
            "total_transcript_segments": segment_count,
            "total_host_segments": host_segment_count,
            "classified_or_recorded_host_segments": len(actions),
            "completed_chunks": sum(1 for c in completed_chunks if c.get("status") == "completed"),
            "failed_chunk_count": len(failed_chunks),
            "action_counts": {
                label: sum(1 for a in actions if a.get("host_action") == label)
                for label in ACTION_LABELS
            },
        },
        "chunk_status": {"completed": completed_chunks, "failed": failed_chunks},
        "actions": actions,
    }


def discover_video_dirs(root: Path) -> list[Path]:
    """Discover video-ID folders with the expected direct layout: root/<video_id>/final.json."""
    return sorted(path.parent for path in root.glob("*/final.json") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify host actions from final.json['segments'] using Gemini.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=24, help="Number of transcript segments per core chunk (default: 24)")
    parser.add_argument("--context-segments", type=int, default=4, help="Neighboring segments added around each chunk (default: 4)")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout in seconds per Gemini request (default: 90)")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient failures (default: 2)")
    parser.add_argument("--limit-chunks", type=int, help="Process only the first N chunks per video (pilot/test run)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing host_actions.json files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1 or args.context_segments < 0:
        print("ERROR: --chunk-size must be >= 1 and --context-segments must be >= 0", file=sys.stderr)
        return 2
    if args.limit_chunks is not None and args.limit_chunks < 1:
        print("ERROR: --limit-chunks must be >= 1", file=sys.stderr)
        return 2
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in this terminal environment", file=sys.stderr)
        return 2

    videos_root = VIDEOS_ROOT
    if not videos_root.is_dir():
        print(f"ERROR: VIDEOS_ROOT does not exist: {videos_root}", file=sys.stderr)
        print("Edit VIDEOS_ROOT near the top of generate_host_actions.py to the correct directory.", file=sys.stderr)
        return 2

    video_dirs = discover_video_dirs(videos_root)
    if not video_dirs:
        print(f"ERROR: no final.json files found under {videos_root}", file=sys.stderr)
        return 2

    print(f"Videos root: {videos_root}")
    successes = 0
    for video_dir in video_dirs:
        ok, message = process_video(video_dir, args, api_key)
        print(message)
        successes += int(ok)
    print(f"Finished: {successes}/{len(video_dirs)} video(s) processed.")
    return 0 if successes == len(video_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())