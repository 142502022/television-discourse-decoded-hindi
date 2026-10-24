#!/usr/bin/env python3
"""Classify overlap follow-up speech events into a separate JSON artifact.

Hybrid approach:
  1. Conservative deterministic rules propose obvious labels.
  2. Optional Gemini LLM classification uses the complete local window context.
  3. Structural and enum validation runs before the output is written.

This script never modifies the source extraction JSON or final.json.

Examples:
  # Produce a conservative rules-only draft (no API key needed)
  python classify_overlap_followup.py \
      --input data/v2/videos/eiY9KJ2SHnA/overlap_followup_analysis.json \
      --output data/v2/videos/eiY9KJ2SHnA/overlap_followup_classified.json \
      --rules-only

  # Run hybrid classification with Gemini (set GEMINI_API_KEY first)
  python classify_overlap_followup.py \
      --input data/v2/videos/eiY9KJ2SHnA/overlap_followup_analysis.json \
      --output data/v2/videos/eiY9KJ2SHnA/overlap_followup_classified.json \
      --limit-overlaps 10

The rules-only output is a draft, not a human-verified annotation set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/v2/videos/eiY9KJ2SHnA/overlap_followup_analysis.json")
DEFAULT_MODEL = "gemini-2.5-flash"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

UTTERANCE_FORMS = {"question", "statement", "imperative", "fragment", "unclear"}
CONVERSATIONAL_FUNCTIONS = {
    "asking", "challenging", "answering", "interrupting", "yielding_floor",
    "redirecting", "backchanneling", "asserting", "unclear",
}
HOST_BEHAVIORS = {
    "regaining_control", "challenging_participant", "asking_for_clarification",
    "redirecting_topic", "allocating_floor", "allowing_guest_to_continue",
    "summarizing", "closing_discussion", "other", "unclear", "not_applicable",
}
CLASSIFICATION_STATUSES = {
    "pending", "rule_proposed", "llm_proposed", "human_verified", "needs_review", "out_of_scope"
}

SYSTEM_INSTRUCTIONS = r"""You annotate conversational behavior in a televised debate transcript.
Use only the supplied transcript, speaker labels, timestamps, and local context. Do not use outside knowledge, infer real-world identities, invent words, or verify factual claims.

For each speech event, assign exactly one utterance_form:
question, statement, imperative, fragment, unclear.
Assign exactly one primary conversational_function:
asking, challenging, answering, interrupting, yielding_floor, redirecting, backchanneling, asserting, unclear.

WORKING RULE for asking vs challenging:
- Use challenging when the wording or context indicates that the speaker contests, scrutinizes, presses, or demands justification for another participant's claim.
- A request for explanation alone defaults to asking unless there is evidence of contestation or pressure.
- A question is not automatically asking; grammatical form and conversational function are separate.
- A statement can function as challenging, answering, or asserting. Do not automatically label all statements as asserting.
- Use asserting when the speaker presents a claim, position, opinion, or purported fact and no clearer function is supported. This does not mean the claim is factually verified.

For target_speaker, use a speaker ID only if the words and context support who is being addressed. Otherwise use null.
For each event, provide evidence as short exact quotes from that event's source text and a concise reason. Confidence must be between 0 and 1, or null if the judgment is unreliable.

Also classify window-level host_behavior with one of:
regaining_control, challenging_participant, asking_for_clarification, redirecting_topic,
allocating_floor, allowing_guest_to_continue, summarizing, closing_discussion, other,
unclear, not_applicable.
Do not infer regaining_control merely because the host speaks first or speaks after an overlap. Timing alone is not proof of intent. If the host does not speak in the window, use not_applicable. If host speech exists but behavior is not clear, use unclear.

This pilot is English-only. If text appears non-English, mixed-language, or language-uncertain, mark the event needs_review or out_of_scope rather than forcing an English label.

Return one JSON object only with keys:
{
  "speech_events": [
    {
      "event_index": integer,
      "utterance_form": string,
      "conversational_function": string,
      "target_speaker": string or null,
      "classification_evidence": [{"quote": string, "reason": string}],
      "classification_confidence": number or null,
      "classification_status": "llm_proposed" or "needs_review" or "out_of_scope"
    }
  ],
  "interpretation": {
    "host_behavior": string,
    "behavior_evidence": [{"speaker": string, "quote": string, "start": number, "end": number, "reason": string}],
    "confidence": number or null,
    "rationale": string or null,
    "manual_review_status": "pending",
    "reviewer_notes": null
  }
}
Include every input event exactly once, preserving its event_index. Use only allowed labels. Do not include markdown fences."""


def compact_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def deterministic_form(text: str) -> tuple[str | None, str | None]:
    """Return a high-precision form proposal and reason; abstain if ambiguous."""
    stripped = text.strip()
    if not stripped:
        return "unclear", "Transcript text is empty."

    # Transcript punctuation is useful but not definitive because it may be ASR-generated.
    if re.search(r"\?\s*$", stripped):
        return "question", "Transcript ends with question punctuation; punctuation may be ASR-generated."

    words = re.findall(r"[A-Za-z']+", stripped.lower())
    if words:
        imperative_starts = {
            "clarify", "explain", "answer", "listen", "look", "tell", "let", "please",
            "stop", "wait", "go", "come", "consider", "remember", "imagine", "say",
        }
        normalized_start = " ".join(words[:2])
        common_directives = {"just clarify", "please clarify", "please explain", "let me", "let him", "let her"}
        if words[0] in imperative_starts or normalized_start in common_directives:
            return "imperative", "The utterance begins with a likely directive/request form; verify in context."

    # Avoid pretending a declarative transcript is always a complete statement.
    if len(words) >= 4 and stripped[-1:] in {".", "!"}:
        return "statement", "Declarative punctuation and length suggest a statement; transcript punctuation may be imperfect."
    if len(words) <= 3:
        return "fragment", "Short utterance; form may be a fragment or a brief complete utterance."
    return None, None


def deterministic_function(text: str, form: str | None) -> tuple[str | None, str | None]:
    """Only propose conversational functions for very obvious lexical patterns."""
    low = re.sub(r"\s+", " ", text.lower()).strip(" \t\r\n.,!?\"'")
    if not low:
        return "unclear", "No transcript text is available."

    backchannels = {
        "yes", "yeah", "yep", "right", "correct", "okay", "ok", "mm-hmm", "uh-huh", "exactly"
    }
    if low in backchannels:
        return "backchanneling", "Short acknowledgement token with no clear substantive turn."

    floor_phrases = (
        "go ahead", "let her speak", "let him speak", "let them speak", "let her answer",
        "let him answer", "you may answer", "your turn", "you have the floor",
    )
    if any(phrase in low for phrase in floor_phrases):
        return "yielding_floor", "Explicit phrase appears to hand the floor to another participant."

    closing_phrases = ("thanks for watching", "thank you for watching", "stay safe", "jai hind", "namaskar")
    if any(phrase in low for phrase in closing_phrases):
        return "asserting", "This is a statement-like closing utterance; window-level closing behavior needs context."

    # Do not auto-label a question as asking/challenging; that distinction is contextual.
    if form == "statement":
        return None, None
    return None, None


def clamp_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, number)), 3)


def validate_event_classification(result: dict[str, Any], input_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate LLM output and normalize it; invalid labels become needs_review."""
    by_index: dict[int, dict[str, Any]] = {}
    raw_events = result.get("speech_events", [])
    if not isinstance(raw_events, list):
        raw_events = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("event_index"))
        except (TypeError, ValueError):
            continue
        if index not in by_index:
            by_index[index] = item

    normalized = []
    for index, source_event in enumerate(input_events):
        item = by_index.get(index, {})
        form = item.get("utterance_form")
        function = item.get("conversational_function")
        status = item.get("classification_status", "needs_review")
        if form not in UTTERANCE_FORMS:
            form = "unclear"
            status = "needs_review"
        if function not in CONVERSATIONAL_FUNCTIONS:
            function = "unclear"
            status = "needs_review"
        if status not in {"llm_proposed", "needs_review", "out_of_scope"}:
            status = "needs_review"

        target = item.get("target_speaker")
        if target is not None and not isinstance(target, str):
            target = None
        evidence = item.get("classification_evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        cleaned_evidence = []
        source_text = compact_text(source_event.get("text"))
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            quote = compact_text(entry.get("quote"))
            reason = compact_text(entry.get("reason"))
            if quote and quote not in source_text:
                # Keep the output auditable; do not retain fabricated/non-verbatim quotes.
                status = "needs_review"
                continue
            if quote or reason:
                cleaned_evidence.append({"quote": quote, "reason": reason})

        normalized.append({
            "event_index": index,
            "utterance_form": form,
            "conversational_function": function,
            "target_speaker": target,
            "classification_evidence": cleaned_evidence,
            "classification_confidence": clamp_confidence(item.get("classification_confidence")),
            "classification_status": status,
        })
    return normalized


def validate_interpretation(result: dict[str, Any], overlap_record: dict[str, Any], host_speaker: str) -> dict[str, Any]:
    raw = result.get("interpretation", {})
    if not isinstance(raw, dict):
        raw = {}
    behavior = raw.get("host_behavior")
    if behavior not in HOST_BEHAVIORS:
        behavior = "unclear"
    if not overlap_record.get("observations", {}).get("host_spoke_in_window", False):
        behavior = "not_applicable"

    evidence = raw.get("behavior_evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    events = overlap_record.get("speech_events", [])
    valid_texts = {(event.get("speaker"), event.get("text")) for event in events}
    cleaned = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        quote = compact_text(item.get("quote"))
        if quote and (speaker, quote) not in valid_texts:
            # Require a verbatim quote from an event attributed to this speaker.
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "speaker": speaker if isinstance(speaker, str) else host_speaker,
            "quote": quote,
            "start": start,
            "end": end,
            "reason": compact_text(item.get("reason")),
        })

    confidence = clamp_confidence(raw.get("confidence"))
    rationale = raw.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = None
    if behavior in {"unclear", "not_applicable"}:
        confidence = None if behavior == "not_applicable" else confidence

    return {
        "host_behavior": behavior,
        "behavior_evidence": cleaned,
        "confidence": confidence,
        "rationale": rationale,
        "manual_review_status": "pending",
        "reviewer_notes": None,
    }


def call_gemini(api_key: str, model: str, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    url = API_ENDPOINT.format(model=model, key=api_key)
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
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            text_parts = []
            for candidate in response_data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            if not text_parts:
                raise ValueError("Gemini response contained no text candidate.")
            parsed = json.loads("\n".join(text_parts))
            if not isinstance(parsed, dict):
                raise ValueError("Gemini response JSON was not an object.")
            return parsed
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request failed after 3 attempts: {last_error}")


def build_llm_payload(record: dict[str, Any], host_speaker: str) -> dict[str, Any]:
    events = []
    for index, event in enumerate(record.get("speech_events", [])):
        form_candidate, form_reason = deterministic_form(compact_text(event.get("text")))
        function_candidate, function_reason = deterministic_function(compact_text(event.get("text")), form_candidate)
        events.append({
            "event_index": index,
            "speaker": event.get("speaker"),
            "start": event.get("start"),
            "end": event.get("end"),
            "text": event.get("text"),
            "starts_in_window": event.get("starts_in_window"),
            "active_at_window_start": event.get("active_at_window_start"),
            "extends_beyond_window": event.get("extends_beyond_window"),
            "deterministic_candidates": {
                "utterance_form": form_candidate,
                "utterance_form_reason": form_reason,
                "conversational_function": function_candidate,
                "conversational_function_reason": function_reason,
            },
        })
    return {
        "overlap_id": record.get("overlap_id"),
        "overlap": record.get("overlap"),
        "window_start": record.get("window_start"),
        "window_end": record.get("window_end"),
        "host_speaker_id": host_speaker,
        "observations": record.get("observations", {}),
        "speech_events": events,
    }


def classify_rules_only(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classified = []
    for index, event in enumerate(record.get("speech_events", [])):
        text = compact_text(event.get("text"))
        form, form_reason = deterministic_form(text)
        function, function_reason = deterministic_function(text, form)
        status = "rule_proposed" if form or function else "needs_review"
        evidence = []
        if text and (form_reason or function_reason):
            # Exact source text is retained as evidence; no rewritten quote is introduced.
            evidence.append({
                "quote": text,
                "reason": "; ".join(reason for reason in (form_reason, function_reason) if reason),
            })
        classified.append({
            "event_index": index,
            "utterance_form": form or "unclear",
            "conversational_function": function or "unclear",
            "target_speaker": None,
            "classification_evidence": evidence,
            "classification_confidence": 0.55 if form else None,
            "classification_status": status if (form and function) else "needs_review",
        })

    host_speaks = bool(record.get("observations", {}).get("host_spoke_in_window", False))
    interpretation = {
        "host_behavior": "unclear" if host_speaks else "not_applicable",
        "behavior_evidence": [],
        "confidence": None,
        "rationale": "Rules-only mode does not infer host behavior without a high-precision explicit cue.",
        "manual_review_status": "pending",
        "reviewer_notes": None,
    }
    return classified, interpretation


def adapt_source_record(record: dict[str, Any], host_speaker: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Adapt either the older classifier schema or overlap-followup-analysis-1.1.

    Returns an internal classifier record and the original speech-event list, so
    classifications can be written back without changing extraction fields.
    """
    if isinstance(record.get("post_overlap_speaker_sequence"), list):
        source_events = record["post_overlap_speaker_sequence"]
        window = record.get("observation_window", {})
        normalized_events = []
        for event in source_events:
            normalized_events.append({
                "speaker": event.get("speaker"),
                "start": event.get("window_intersection_start", event.get("segment_start")),
                "end": event.get("window_intersection_end", event.get("segment_end")),
                "text": event.get("text", ""),
                "starts_in_window": not bool(event.get("started_during_overlap", False)),
                "active_at_window_start": bool(event.get("continues_from_overlap", False)),
                "extends_beyond_window": False,
            })
        internal = {
            "overlap_id": record.get("event_id"),
            "overlap": {
                "start": record.get("overlap_start"),
                "end": record.get("overlap_end"),
                "duration_s": record.get("overlap_duration_s"),
                "speakers": record.get("overlapping_speakers", []),
            },
            "window_start": window.get("start"),
            "window_end": window.get("end"),
            "observations": {
                "host_spoke_in_window": record.get("host_spoke_in_window", False),
                "host_participated_in_overlap": record.get("host_participated_in_overlap", False),
                "first_speaker_present_in_window": record.get("first_speaker_present_in_window"),
                "speakers_present_in_window": record.get("speakers_present_in_window", []),
                "host_continued_from_overlap_into_window": record.get("host_continued_from_overlap_into_window", False),
                "speaker_after_host": record.get("speaker_after_host", []),
            },
            "speech_events": normalized_events,
        }
        return internal, source_events

    source_events = record.get("speech_events", [])
    if not isinstance(source_events, list):
        source_events = []
    return record, source_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to overlap_followup_analysis.json")
    parser.add_argument("--output", type=Path, default=None, help="Output path; defaults beside input")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--host-speaker", default=None, help="Override host speaker ID; otherwise read source metadata")
    parser.add_argument("--limit-overlaps", type=int, default=None, help="Classify only the first N overlap windows (pilot mode)")
    parser.add_argument("--rules-only", action="store_true", help="Skip Gemini and emit conservative rule proposals")
    parser.add_argument("--pause-seconds", type=float, default=0.25, help="Pause between Gemini requests")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input.resolve()}", file=sys.stderr)
        return 2
    if args.limit_overlaps is not None and args.limit_overlaps < 1:
        print("ERROR: --limit-overlaps must be >= 1", file=sys.stderr)
        return 2

    with args.input.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict):
        print("ERROR: input must contain a JSON object.", file=sys.stderr)
        return 2
    if isinstance(source.get("overlap_events"), list):
        record_key = "overlap_events"
    elif source.get("schema_version") == "overlap-followup-analysis-1.1" and isinstance(source.get("events"), list):
        record_key = "events"
    else:
        print("ERROR: input must contain either an overlap_events list or events list from overlap-followup-analysis-1.1.", file=sys.stderr)
        return 2

    host_speaker = args.host_speaker or source.get("host_speaker_id") or source.get("source", {}).get("host_speaker") or "SPEAKER_02"
    api_key = os.environ.get("GEMINI_API_KEY")
    use_llm = not args.rules_only and bool(api_key)
    if not args.rules_only and not api_key:
        print("WARNING: GEMINI_API_KEY is not set; falling back to conservative rules-only mode.", file=sys.stderr)
        print("         Set GEMINI_API_KEY to enable the LLM stage.", file=sys.stderr)

    output_data = deepcopy(source)
    output_data["schema_version"] = "overlap-followup-classified-1.0"
    output_data["analysis_status"] = "hybrid_classification_completed_with_pending_human_review" if use_llm else "rules_only_draft_pending_human_review"
    output_data["classification_run"] = {
        "method": "hybrid_rules_plus_gemini" if use_llm else "deterministic_rules_only",
        "model": args.model if use_llm else None,
        "host_speaker": host_speaker,
        "pilot_limit": args.limit_overlaps,
        "working_rule_asking_vs_challenging": (
            "Use challenging when wording or context contests, scrutinizes, presses, or demands justification for a claim. "
            "A request for explanation alone defaults to asking unless there is evidence of contestation or pressure."
        ),
        "human_verified": False,
    }

    records = output_data[record_key]
    process_count = min(len(records), args.limit_overlaps) if args.limit_overlaps is not None else len(records)
    for record_index, record in enumerate(records):
        internal_record, original_events = adapt_source_record(record, host_speaker)
        if record_index >= process_count:
            # Mark only the classification status; retain extraction data unchanged.
            for event in original_events:
                event["classification_status"] = "pending"
            record.setdefault("interpretation", {})["manual_review_status"] = "pending"
            continue

        normalized_events = internal_record.get("speech_events", [])
        if use_llm:
            try:
                llm_result = call_gemini(
                    api_key=api_key,
                    model=args.model,
                    payload=build_llm_payload(internal_record, host_speaker),
                )
                normalized = validate_event_classification(llm_result, normalized_events)
                interpretation = validate_interpretation(llm_result, internal_record, host_speaker)
                method = "llm"
            except Exception as exc:  # Preserve the run and mark this window for review.
                print(f"WARNING: overlap {record.get('event_id', record.get('overlap_id'))} failed LLM classification: {exc}", file=sys.stderr)
                normalized, interpretation = classify_rules_only(internal_record)
                for event in normalized:
                    event["classification_status"] = "needs_review"
                interpretation["host_behavior"] = (
                    "unclear" if internal_record.get("observations", {}).get("host_spoke_in_window") else "not_applicable"
                )
                interpretation["rationale"] = f"LLM call failed; conservative rules-only fallback. Details: {exc}"
                method = "rules_fallback_after_llm_error"
        else:
            normalized, interpretation = classify_rules_only(internal_record)
            method = "rules_only"

        # Copy only classification fields back; preserve source extraction fields verbatim.
        for event, classification in zip(original_events, normalized):
            event["utterance_form"] = classification["utterance_form"]
            event["conversational_function"] = classification["conversational_function"]
            event["target_speaker"] = classification["target_speaker"]
            event["classification_evidence"] = classification["classification_evidence"]
            event["classification_confidence"] = classification["classification_confidence"]
            event["classification_status"] = classification["classification_status"]
            event["classification_method"] = method
        record["interpretation"] = interpretation
        record["interpretation"]["classification_method"] = method

        if use_llm and args.pause_seconds > 0 and record_index + 1 < process_count:
            time.sleep(args.pause_seconds)

    output = args.output or args.input.with_name("overlap_followup_classified.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    classified_events = sum(
        1 for record in records[:process_count]
        for event in (record.get("post_overlap_speaker_sequence", []) if isinstance(record.get("post_overlap_speaker_sequence"), list) else record.get("speech_events", []))
        if event.get("classification_status") in {"rule_proposed", "llm_proposed", "needs_review", "out_of_scope"}
    )
    print(f"Input overlap windows: {len(records)}")
    print(f"Windows processed: {process_count}")
    print(f"Speech events with classification output/status: {classified_events}")
    print(f"Method: {'hybrid rules + Gemini' if use_llm else 'conservative rules-only draft'}")
    print(f"Human verified: no")
    print(f"Output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())