#!/usr/bin/env python3
"""
Classify host actions after overlap events using Gemini and the 3-second window.

Input for each video:
  data/v2/videos/<video_id>/overlap_followup_analysis.json
  data/v2/videos/<video_id>/host_identification/host_identification.json
  (or host_identification.json directly in the video directory)

Output:
  data/v2/videos/<video_id>/interruption_management.json

Required environment variable:
  GEMINI_API_KEY

Recommended first run (small pilot):
  python generate_interruption_management_v2.py --limit-overlaps 5

Run all events:
  python generate_interruption_management_v2.py

Overwrite existing outputs:
  python generate_interruption_management_v2.py --overwrite

This script does not modify the input JSON files. It preserves existing event IDs.
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

DEFAULT_VIDEOS_ROOT = Path("data/v2/videos")
INPUT_FILENAME = "*/overlap_followup_analysis.json"
OUTPUT_FILENAME = DEFAULT_VIDEOS_ROOT /"interruption_management.json"
DEFAULT_MODEL = "gemini-3.5-flash"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

HOST_ACTIONS = {
    "regaining_control",
    "challenging_participant",
    "allocating_floor",
    "allowing_guest_to_continue",
    "redirecting_topic",
    "other_or_unclear",
    "not_applicable",
}

SYSTEM_INSTRUCTIONS = r"""
You classify the host's conversational action in a televised debate immediately
after an overlap. Use ONLY the supplied overlap metadata, host identity, timestamps,
speaker sequence, and verbatim transcript text. Do not use outside knowledge or
invent words. The observation window is the 3-second interval after overlap_end.
The supplied speaker sequence includes any speech segment intersecting that
window, including a segment that began during the overlap and continued into it.

Assign exactly one host_action:
- regaining_control: the host actively re-establishes order or control of the floor,
  e.g. asks participants to stop/take turns, reasserts the moderator's turn, or
  manages competing speakers. Do NOT infer this merely because the host speaks
  first or speaks after an overlap.
- challenging_participant: the host contests, scrutinizes, presses, or demands
  justification for a participant's claim/position. A question alone is not enough;
  its wording/context must show challenge or pressure. A neutral request for
  explanation is not automatically challenging.
- allocating_floor: the host explicitly gives or directs the turn to a participant,
  invites them to answer/respond, or names who should speak next.
- allowing_guest_to_continue: the host explicitly invites/permits a guest to
  continue or finish their point (e.g. "go on", "continue", "finish your point").
  Do not use this merely because a guest is speaking while the host is silent.
- redirecting_topic: the host explicitly redirects the discussion to a different
  question, subject, or agenda item. Do not infer this from topic differences alone.
- other_or_unclear: the host speaks in the window but available words/context do
  not support one of the above actions with reasonable confidence, or cues conflict.
- not_applicable: use ONLY when the source confirms the host does not speak in
  the post-overlap window. If host speech status is unknown, do not use this label.

Use surrounding turns in the supplied 3-second window to interpret the host's
words. Host action may be inferred from a short utterance when its function is
clear, but do not invent missing context. If the host's utterance continues from
the overlap, it still counts as host speech in the window, but timing alone does
not prove its function.

Return JSON only with:
{
  "host_action": one allowed label,
  "confidence": number from 0 to 1 or null,
  "rationale": short explanation,
  "classification_evidence": [
    {"speaker": "speaker ID", "quote": "exact verbatim quote from supplied text",
     "start": number or null, "end": number or null,
     "reason": "why this quote supports the label"}
  ],
  "review_status": "pending" or "needs_review"
}
Evidence quotes must be exact substrings of supplied transcript text. Use evidence
from the host when host speech exists. For not_applicable, evidence may be empty.
If confidence is below 0.65, use other_or_unclear unless the action is not_applicable.
Set review_status to needs_review for other_or_unclear or confidence below 0.75;
otherwise pending. Never mark human_verified.
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def first_string(obj: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_host_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    value = first_string(data, (
        "predicted_host_id", "predicted_host", "host_speaker_id", "host_id", "predicted_label"
    ))
    if value:
        return value
    for key in ("prediction", "host", "result", "host_identification"):
        value = first_string(data.get(key), (
            "predicted_host_id", "predicted_host", "host_speaker_id", "host_id",
            "predicted_label", "speaker_id", "speaker", "label"
        ))
        if value:
            return value
    for key in ("top_candidates", "candidates", "host_candidates"):
        candidates = data.get(key)
        if not isinstance(candidates, list):
            continue
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (
                0 if isinstance(pair[1], dict) and pair[1].get("is_predicted_host") is True else 1,
                0 if isinstance(pair[1], dict) and pair[1].get("rank") == 1 else 1,
                pair[0],
            ),
        )
        for _, candidate in ranked:
            value = first_string(candidate, ("speaker_id", "speaker", "label", "candidate", "host_id"))
            if value:
                return value
    return None


def locate_host_file(video_dir: Path) -> Path | None:
    for relative in (
        Path("host_identification/host_identification.json"),
        Path("host_identification.json"),
    ):
        candidate = video_dir / relative
        if candidate.is_file():
            return candidate
    folder = video_dir / "host_identification"
    if folder.is_dir():
        for candidate in sorted(folder.glob("*.json")):
            if "host" in candidate.name.lower() and "identification" in candidate.name.lower():
                return candidate
    return None


def resolve_host(video_dir: Path, source: dict[str, Any]) -> tuple[str, str | None]:
    host_file = locate_host_file(video_dir)
    if host_file:
        host_id = extract_host_id(load_json(host_file))
        if not host_id:
            raise ValueError(f"Could not extract host speaker ID from {host_file}")
        return host_id, str(host_file)

    # The overlap analysis can be used as a compatibility fallback if the
    # dedicated host-identification artifact is absent.
    host_id = first_string(source, ("host_speaker_id",))
    if host_id:
        return host_id, None
    raise ValueError(f"No usable host ID or host_identification.json found in {video_dir}")


def get_events(source: dict[str, Any]) -> list[dict[str, Any]]:
    events = source.get("events", source.get("overlap_events", []))
    if not isinstance(events, list):
        raise ValueError("Input must contain an events or overlap_events list.")
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("Every event must be a JSON object.")
    return events


def get_sequence(event: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = event.get("post_overlap_speaker_sequence", [])
    return [item for item in sequence if isinstance(item, dict)] if isinstance(sequence, list) else []


def confirmed_host_spoke(event: dict[str, Any], host_id: str) -> bool:
    explicit = event.get("host_spoke_in_window")
    if isinstance(explicit, bool):
        return explicit
    return any(segment.get("speaker") == host_id for segment in get_sequence(event))


def prepare_payload(event: dict[str, Any], host_id: str) -> dict[str, Any]:
    sequence = []
    for segment in get_sequence(event):
        sequence.append({
            "speaker": segment.get("speaker"),
            "start": segment.get("window_intersection_start", segment.get("segment_start")),
            "end": segment.get("window_intersection_end", segment.get("segment_end")),
            "segment_start": segment.get("segment_start"),
            "segment_end": segment.get("segment_end"),
            "relative_start_s": segment.get("relative_start_s"),
            "relative_end_s": segment.get("relative_end_s"),
            "started_during_overlap": segment.get("started_during_overlap"),
            "continues_from_overlap": segment.get("continues_from_overlap"),
            "text": segment.get("text", ""),
        })
    return {
        "event_id": event.get("event_id"),
        "host_speaker_id": host_id,
        "overlap_start": event.get("overlap_start"),
        "overlap_end": event.get("overlap_end"),
        "overlap_duration_s": event.get("overlap_duration_s"),
        "overlapping_speakers": event.get("overlapping_speakers", []),
        "host_participated_in_overlap": event.get("host_participated_in_overlap"),
        "host_spoke_in_window": confirmed_host_spoke(event, host_id),
        "observation_window": event.get("observation_window", {"duration_s": 3.0}),
        "first_speaker_present_in_window": event.get("first_speaker_present_in_window"),
        "speaker_sequence_in_post_overlap_window": sequence,
    }


def call_gemini(api_key: str, model: str, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    url = API_ENDPOINT.format(model=model)
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTIONS}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            texts = []
            for candidate in response_data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if isinstance(part.get("text"), str):
                        texts.append(part["text"])
            if not texts:
                raise ValueError("API response had no text candidate.")
            result = json.loads("\n".join(texts))
            if not isinstance(result, dict):
                raise ValueError("API response was not a JSON object.")
            return result
        except urllib.error.HTTPError as exc:
            # Preserve Google's response body; it usually explains a 404/400 clearly.
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail[:1500]}")
            # A 4xx response will not improve by retrying the same request.
            if 400 <= exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request failed after 3 attempts: {last_error}")


def validate_result(result: dict[str, Any], event: dict[str, Any], host_id: str) -> dict[str, Any]:
    host_spoke = confirmed_host_spoke(event, host_id)
    if not host_spoke:
        return {
            "host_action": "not_applicable",
            "confidence": 1.0,
            "rationale": "Source confirms the host does not speak in the 3-second post-overlap window.",
            "classification_evidence": [],
            "review_status": "pending",
        }

    action = result.get("host_action")
    if action not in HOST_ACTIONS or action == "not_applicable":
        action = "other_or_unclear"

    try:
        confidence = float(result.get("confidence"))
        confidence = min(1.0, max(0.0, confidence))
    except (TypeError, ValueError):
        confidence = None

    if confidence is None or confidence < 0.65:
        action = "other_or_unclear"

    # Validate quotes against verbatim text from the supplied window.
    source_texts = [
        str(segment.get("text", ""))
        for segment in get_sequence(event)
        if isinstance(segment.get("text"), str)
    ]
    evidence = result.get("classification_evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    clean_evidence = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        quote = item.get("quote")
        speaker = item.get("speaker")
        if not isinstance(quote, str) or not quote:
            continue
        if not any(quote in text for text in source_texts):
            continue
        if speaker != host_id:
            # Evidence for host_action must come from the host, not a guest quote.
            continue
        start, end = item.get("start"), item.get("end")
        clean_evidence.append({
            "speaker": host_id,
            "quote": quote,
            "start": start if isinstance(start, (int, float)) else None,
            "end": end if isinstance(end, (int, float)) else None,
            "reason": str(item.get("reason", "")),
        })

    if action != "other_or_unclear" and not clean_evidence:
        action = "other_or_unclear"

    rationale = result.get("rationale")
    if not isinstance(rationale, str):
        rationale = "No rationale returned."
    review_status = result.get("review_status")
    if action == "other_or_unclear" or (confidence is not None and confidence < 0.75):
        review_status = "needs_review"
    elif review_status not in {"pending", "needs_review"}:
        review_status = "pending"

    return {
        "host_action": action,
        "confidence": confidence,
        "rationale": rationale,
        "classification_evidence": clean_evidence,
        "review_status": review_status,
    }


def classify_event(event: dict[str, Any], host_id: str, api_key: str, model: str) -> dict[str, Any]:
    if not confirmed_host_spoke(event, host_id):
        return {
            "host_action": "not_applicable",
            "confidence": 1.0,
            "rationale": "Source confirms the host does not speak in the 3-second post-overlap window.",
            "classification_evidence": [],
            "review_status": "pending",
        }
    payload = prepare_payload(event, host_id)
    result = call_gemini(api_key, model, payload)
    return validate_result(result, event, host_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--input-name", default=INPUT_FILENAME)
    parser.add_argument("--output-name", default=OUTPUT_FILENAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-overlaps", type=int, default=None,
                        help="Process only the first N events per video (recommended for initial pilot).")
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set. Set it before running this classifier.", file=sys.stderr)
        return 2
    if not args.videos_root.is_dir():
        print(f"ERROR: videos root not found: {args.videos_root.resolve()}", file=sys.stderr)
        return 2
    if args.limit_overlaps is not None and args.limit_overlaps < 1:
        print("ERROR: --limit-overlaps must be >= 1", file=sys.stderr)
        return 2

    inputs = sorted(args.videos_root.glob(f"*/{args.input_name}"))
    if not inputs:
        print(f"No {args.input_name} files found under {args.videos_root.resolve()}", file=sys.stderr)
        return 1

    generated = skipped = failed = 0
    for input_path in inputs:
        video_dir = input_path.parent
        output_path = video_dir / args.output_name
        if output_path.exists() and not args.overwrite:
            print(f"SKIP: {output_path} exists (use --overwrite to replace).")
            skipped += 1
            continue

        try:
            source = load_json(input_path)
            if not isinstance(source, dict):
                raise ValueError("Input JSON root must be an object.")
            host_id, host_file = resolve_host(video_dir, source)
            events = get_events(source)
            process_count = min(len(events), args.limit_overlaps) if args.limit_overlaps else len(events)
            records = []
            counts: dict[str, int] = {}

            for index, event in enumerate(events):
                if index < process_count:
                    try:
                        classification = classify_event(event, host_id, api_key, args.model)
                    except Exception as exc:
                        print(
                            f"WARNING {video_dir.name} event {event.get('event_id')}: {exc}",
                            file=sys.stderr,
                        )
                        classification = {
                            "host_action": "other_or_unclear" if confirmed_host_spoke(event, host_id) else "not_applicable",
                            "confidence": None if confirmed_host_spoke(event, host_id) else 1.0,
                            "rationale": f"Classifier request failed; review required. {exc}",
                            "classification_evidence": [],
                            "review_status": "needs_review",
                        }
                    if args.pause_seconds > 0 and index + 1 < process_count:
                        time.sleep(args.pause_seconds)
                else:
                    classification = {
                        "host_action": "other_or_unclear" if confirmed_host_spoke(event, host_id) else "not_applicable",
                        "confidence": None if confirmed_host_spoke(event, host_id) else 1.0,
                        "rationale": "Not processed because --limit-overlaps was used.",
                        "classification_evidence": [],
                        "review_status": "needs_review",
                    }

                host_segments = [
                    segment for segment in get_sequence(event)
                    if segment.get("speaker") == host_id and isinstance(segment.get("text"), str)
                ]
                host_evidence = " | ".join(
                    f"[{segment.get('segment_start')}–{segment.get('segment_end')}] {segment['text'].strip()}"
                    for segment in host_segments if segment.get("text", "").strip()
                )
                video = source.get("video") if isinstance(source.get("video"), dict) else {}
                video_id = str(video.get("video_id") or video_dir.name)

                duration = event.get("overlap_duration_s")
                if duration is None:
                    start, end = event.get("overlap_start"), event.get("overlap_end")
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        duration = round(end - start, 3)

                first_speaker = event.get("first_speaker_present_in_window")
                if first_speaker is None and get_sequence(event):
                    first_speaker = get_sequence(event)[0].get("speaker")

                record = {
                    "video_id": video_id,
                    "event_id": event.get("event_id"),
                    "overlap_duration_s": duration,
                    "overlapping_speakers": event.get("overlapping_speakers", []),
                    "host_participated_in_overlap": event.get("host_participated_in_overlap"),
                    "host_spoke_after_overlap": confirmed_host_spoke(event, host_id),
                    "first_speaker_after_overlap": first_speaker,
                    "host_action": classification["host_action"],
                    "evidence": host_evidence,
                    "classification_evidence": classification["classification_evidence"],
                    "classification_rationale": classification["rationale"],
                    "confidence": classification["confidence"],
                    "review_status": classification["review_status"],
                }
                records.append(record)
                counts[record["host_action"]] = counts.get(record["host_action"], 0) + 1

            output = {
                "schema_version": "interruption-management-1.3",
                "video": {
                    "video_id": str((source.get("video") or {}).get("video_id") or video_dir.name)
                    if isinstance(source.get("video"), dict) else video_dir.name,
                    "title": source.get("video", {}).get("title")
                    if isinstance(source.get("video"), dict) else None,
                },
                "host_speaker_id": host_id,
                "host_identification_file": host_file,
                "source_file": input_path.name,
                "observation_window_s": source.get("window_seconds", 3.0),
                "classification_method": "Gemini transcript-and-context classification with evidence validation",
                "classification_model": args.model,
                "classification_taxonomy": {
                    "host_action": [
                        "regaining_control", "challenging_participant", "allocating_floor",
                        "allowing_guest_to_continue", "redirecting_topic",
                        "other_or_unclear", "not_applicable",
                    ],
                    "review_status": ["pending", "needs_review", "human_verified"],
                },
                "summary": {
                    "num_overlap_events": len(records),
                    "events_processed_by_model": process_count,
                    "host_action_counts": counts,
                    "events_flagged_for_review": sum(r["review_status"] == "needs_review" for r in records),
                },
                "interpretation_note": (
                    "Host actions are model-proposed from the host's verbatim transcript and the "
                    "post-overlap speaker sequence. They are not human-verified. Timing or first-speaker "
                    "status alone is never sufficient to infer intent. other_or_unclear flags cases "
                    "needing context review and may help identify events for a longer observation window. "
                    "When --limit-overlaps is used, unprocessed events are marked needs_review."
                ),
                "events": records,
            }
            write_json(output_path, output)
            print(f"OK {video_dir.name}: {len(records)} events -> {output_path}; counts={counts}")
            generated += 1
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"ERROR {input_path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Finished: generated={generated}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())