#!/usr/bin/env python3
"""Context-aware agreement/disagreement analysis for English debate transcripts.

Processes threshold_5.0 only. Uses Gemini to classify directed speaker-to-speaker
positions using full transcript turns and nearby conversational context.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VARIANT = "threshold_5.0"
SCRIPT_VERSION = "agreement-disagreement-v2.0-json-recovery"
SCHEMA_VERSION = "agreement-disagreement-2.0"
PROMPT_VERSION = "four-label-two-evidence-context3-v2.0"
RECONSTRUCTION_CONFIG = {"method": "word_midpoint_to_closest_same_speaker_interval", "fallback_gap_seconds": 1.5}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
LABELS = {"agreement", "partial_agreement", "disagreement", "uncertain"}
BATCH_TARGETS = 5
CONTEXT_TURNS = 3

SYSTEM_PROMPT = r"""You analyze a televised debate transcript, which may be Hindi, English, or another language, and identify directed stance interactions. Analyze the original language; do not translate evidence quotations.

For EACH supplied response turn, return exactly one result. Choose the specific earlier turn
whose claim is being addressed. It may be any of the supplied previous context turns, not
necessarily the immediately preceding turn. The responding speaker is the speaker of the
response turn; the target speaker is the speaker of the selected earlier turn. Never reverse
the direction.

Allowed classifications (exactly these four):
- agreement: supports the target statement.
- partial_agreement: supports a meaningful part but qualifies or rejects another part.
- disagreement: rejects, contradicts, or clearly opposes the target statement.
- uncertain: the target or stance is unclear, there is no discernible stance, or the turn is
  merely moderation, a question, clarification, repetition, topic change, or unrelated speech.

Rules:
1. Analyze meaning and conversational context, not keywords, sentiment, or topic similarity.
2. Do not infer a stance from a question, speaker-name mention, overlap, interruption, or
   moderator/host role alone. Host identity is metadata only.
3. Do not assume the immediately previous speaker is the target. Select an earlier target only
   when the response semantically addresses that speaker's specific statement.
4. Return uncertain when target or stance cannot be established confidently. Do not omit it.
5. Evidence MUST contain two exact quotations copied verbatim from the supplied turns:
   statement_1: exact target-speaker claim; statement_2: exact responding-speaker response.
   Do not paraphrase, normalize, repair, or invent quotations. If either quote cannot be
   identified exactly, use uncertain and empty strings for both quotations.
6. reasoning briefly explains the semantic relationship and why this target turn was selected.
7. confidence must be high, medium, or low.
8. If target_turn_id is not the immediately preceding turn, set non_immediate_target=true
   and marker="*". Otherwise set false and marker="". For uncertain with no target use null,
   false, and empty marker.
9. Return one result for every supplied response_turn_id, no duplicates.

Return ONLY valid JSON matching this shape:
{"results":[{"response_turn_id":"T0002","target_turn_id":"T0001",
"target_speaker":"SPEAKER_00","responding_speaker":"SPEAKER_01",
"classification":"agreement|partial_agreement|disagreement|uncertain",
"non_immediate_target":false,"marker":"",
"evidence":{"statement_1":"exact target claim","statement_2":"exact response"},
"reasoning":"brief explanation","confidence":"high|medium|low"}]}
Never return markdown fences or additional text."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_turns(merge: dict[str, Any], max_fallback_gap: float = 1.5) -> list[dict[str, Any]]:
    """Reconstruct turns from word-level speaker labels and audit every source interval."""
    raw_turns = merge.get("speaker_turns") or []
    segments = merge.get("segments") or []
    intervals = []
    for idx, raw in enumerate(raw_turns):
        try:
            st, en = float(raw["start"]), float(raw["end"])
            speaker = str(raw.get("speaker") or raw.get("speaker_id") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        if speaker and en > st:
            intervals.append({"raw_index": idx, "interval_id": f"SI_{idx+1:03d}",
                              "start": st, "end": en, "speaker_id": speaker, "words": []})

    words = []
    asr_word_count = words_with_speaker = 0
    segment_fallback_count = 0
    for seg_idx, seg in enumerate(segments):
        if seg.get("words"):
            for wi, word in enumerate(seg["words"]):
                asr_word_count += 1
                speaker = str(word.get("speaker") or word.get("speaker_id") or "").strip()
                if speaker:
                    words_with_speaker += 1
                try:
                    ws, we = float(word["start"]), float(word["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                txt = normalize_text(word.get("word") or word.get("text"))
                if txt:
                    words.append({"start": ws, "end": we, "speaker_id": speaker, "text": txt,
                                  "segment_index": seg_idx, "word_index": wi})
        else:
            txt = normalize_text(seg.get("text"))
            if txt:
                segment_fallback_count += 1
                try:
                    ws, we = float(seg["start"]), float(seg["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                speaker = str(seg.get("speaker") or seg.get("speaker_id") or "").strip()
                words.append({"start": ws, "end": we, "speaker_id": speaker, "text": txt,
                              "segment_index": seg_idx, "word_index": None})

    assigned_ids = set()
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        candidates = [iv for iv in intervals if iv["speaker_id"] == word["speaker_id"]
                      and iv["start"] <= mid <= iv["end"]]
        if candidates:
            chosen = min(candidates, key=lambda iv: abs(mid - (iv["start"] + iv["end"]) / 2))
            chosen["words"].append(word)
            assigned_ids.add(id(word))

    turns = []
    interval_audit = []
    for iv in intervals:
        iv_words = sorted(iv["words"], key=lambda w: (w["start"], w["end"]))
        text = normalize_text(" ".join(w["text"] for w in iv_words))
        overlapping = [other for other in intervals if other is not iv
                       and min(iv["end"], other["end"]) > max(iv["start"], other["start"])]
        # Words whose midpoint falls inside the interval, irrespective of speaker.
        midpoint_words = [w for w in words if iv["start"] <= (w["start"]+w["end"])/2 <= iv["end"]]
        matching_words = [w for w in midpoint_words if w["speaker_id"] == iv["speaker_id"]]
        words_of_other_speaker = [w for w in midpoint_words if w["speaker_id"] != iv["speaker_id"]]
        if text:
            turns.append({"start": iv["start"], "end": iv["end"], "speaker_id": iv["speaker_id"],
                          "text": text, "word_count": len(iv_words), "turn_source": "speaker_turn_interval"})
        reason = None
        if not text:
            if not midpoint_words:
                reason = "no_word_midpoint_inside_interval"
            elif not matching_words and words_of_other_speaker:
                reason = "interval_contains_word_midpoints_attributed_to_other_speaker"
            else:
                reason = "no_matching_words_assigned_after_interval_matching"
        interval_audit.append({
            "interval_id": iv["interval_id"], "interval_index": iv["raw_index"],
            "speaker_id": iv["speaker_id"], "start": round(iv["start"], 3), "end": round(iv["end"], 3),
            "overlapping_interval_count": len(overlapping),
            "overlapping_intervals": [{"interval_id": x["interval_id"], "speaker_id": x["speaker_id"],
                                       "start": round(x["start"],3), "end": round(x["end"],3)} for x in overlapping],
            "matching_word_count": len(matching_words),
            "matching_words_assigned_to_interval": len(iv_words),
            "other_speaker_word_count_inside_interval": len(words_of_other_speaker),
            "other_speaker_words_inside_interval": [
                {"word": w["text"], "speaker_id": w["speaker_id"],
                 "start": round(w["start"], 3), "end": round(w["end"], 3)}
                for w in words_of_other_speaker
            ],
            "text_recovered": bool(text), "recovered_text": text or None,
            "reason_for_skipping": reason
        })

    # Fallback for word-level material outside all matching intervals.
    unassigned = [w for w in words if id(w) not in assigned_ids and w["speaker_id"]]
    unassigned.sort(key=lambda w: (w["speaker_id"], w["start"], w["end"]))
    groups = []
    for w in unassigned:
        if (not groups or groups[-1]["speaker_id"] != w["speaker_id"]
                or w["start"] - groups[-1]["end"] > max_fallback_gap):
            groups.append({"start": w["start"], "end": w["end"], "speaker_id": w["speaker_id"], "words": [w]})
        else:
            groups[-1]["end"] = max(groups[-1]["end"], w["end"])
            groups[-1]["words"].append(w)
    for g in groups:
        g["words"].sort(key=lambda w: (w["start"], w["end"]))
        turns.append({"start": g["start"], "end": g["end"], "speaker_id": g["speaker_id"],
                      "text": normalize_text(" ".join(w["text"] for w in g["words"])),
                      "word_count": len(g["words"]), "turn_source": "fallback_word_group"})
    turns.sort(key=lambda t: (t["start"], t["end"], t["speaker_id"]))
    for i, turn in enumerate(turns, 1):
        turn["turn_id"] = f"T{i:04d}"
    global TURN_BUILD_DIAGNOSTICS
    TURN_BUILD_DIAGNOSTICS = {
        "raw_speaker_turn_count": len(raw_turns), "valid_speaker_turn_interval_count": len(intervals),
        "asr_word_count": asr_word_count, "words_with_speaker_ids": words_with_speaker,
        "word_items_reconstructed": len(words), "word_items_assigned_to_speaker_turns": len(assigned_ids),
        "word_items_without_matching_interval": len(unassigned), "fallback_turn_count": len(groups),
        "segment_level_fallback_count": segment_fallback_count, "turn_count_reconstructed": len(turns),
        "turns_without_text_skipped": sum(not x["text_recovered"] for x in interval_audit),
        "assignment_policy": "word midpoint assigned to closest matching same-speaker speaker_turn interval",
        "speaker_turn_interval_audit": interval_audit
    }
    if not turns:
        raise ValueError("No transcript turns could be reconstructed")
    return turns

TURN_BUILD_DIAGNOSTICS: dict[str, Any] = {}


def assign_subtopic(timestamp: float, subtopics: list[dict[str, Any]]) -> dict[str, Any] | None:
    for st in subtopics:
        try:
            if float(st["start"]) <= timestamp < float(st["end"]):
                return st
        except (KeyError, TypeError, ValueError):
            continue
    return None


def build_target_batches(turns: list[dict[str, Any]], subtopics: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Each target gets the preceding conversational context; all response turns are examined."""
    targets = []
    for idx, turn in enumerate(turns):
        prior = [p for p in turns[max(0, idx - CONTEXT_TURNS):idx] if p["speaker_id"] != turn["speaker_id"]]
        # A turn with no earlier different speaker cannot express a speaker-to-speaker stance.
        if not prior:
            continue
        st = assign_subtopic(turn["start"], subtopics)
        targets.append({
            "response_turn": turn,
            "prior_turns": prior,
            "immediate_previous_turn_id": turns[idx - 1]["turn_id"] if idx > 0 else None,
            "subtopic": ({"subtopic_id": st.get("subtopic_id"), "title": st.get("title"),
                          "description": st.get("description")} if st else None),
        })
    return [targets[i:i + BATCH_TARGETS] for i in range(0, len(targets), BATCH_TARGETS)]


def extract_json(text: str) -> dict[str, Any]:
    """Parse a complete JSON object; do not use a greedy brace-regex fallback."""
    text = text.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Gemini response must be a JSON object containing a results array")
    return data


def _gemini_request(api_key: str, model: str, system_prompt: str, user_text: str,
                    max_retries: int = 5) -> str:
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(API_URL.format(model=model, key=api_key),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = "".join(part.get("text", "") for candidate in result.get("candidates", [])
                           for part in candidate.get("content", {}).get("parts", []))
            if not text:
                raise ValueError(f"Gemini returned no text: {json.dumps(result)[:1000]}")
            return text
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail[:2000]}") from exc
            wait = min(120, 15 * (2 ** attempt))
            print(f"  Gemini HTTP {exc.code}; retrying in {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini connection failed: {exc}") from exc
            wait = min(120, 15 * (2 ** attempt))
            print(f"  Connection issue; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Gemini request failed after retries")


def call_gemini(api_key: str, model: str, payload: dict[str, Any],
                raw_dump_path: Path | None = None, max_retries: int = 5) -> dict[str, Any]:
    user_text = json.dumps(payload, ensure_ascii=False)
    raw = _gemini_request(api_key, model, SYSTEM_PROMPT, user_text, max_retries)
    try:
        return extract_json(raw)
    except (json.JSONDecodeError, ValueError) as first_error:
        if raw_dump_path:
            raw_dump_path.parent.mkdir(parents=True, exist_ok=True)
            raw_dump_path.write_text(raw, encoding="utf-8")
        print("  Gemini returned malformed JSON; requesting one JSON-only repair", flush=True)
        repair_prompt = ("You are a strict JSON repair tool. Convert the supplied model response "
            "into syntactically valid JSON matching the required schema. Preserve every value, "
            "classification, turn ID, quotation, and reasoning exactly; do not add, remove, "
            "reinterpret, or summarize records. Fix only JSON syntax. Return only JSON.\n\n"
            "Required top-level shape: {\"results\":[...]}\n\nMalformed response:\n" + raw)
        try:
            repaired = _gemini_request(api_key, model,
                "Repair JSON syntax only. Do not change semantic content. Return valid JSON only.",
                repair_prompt, max_retries)
            parsed = extract_json(repaired)
            if raw_dump_path and raw_dump_path.exists():
                raw_dump_path.with_suffix(".repaired.json").write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            return parsed
        except Exception as repair_error:
            if raw_dump_path:
                raw_dump_path.with_suffix(".repair_error.txt").write_text(
                    f"Initial parse error: {first_error}\nRepair error: {repair_error}\n",
                    encoding="utf-8")
            raise ValueError(f"Gemini JSON parse failed and repair failed: {first_error}; {repair_error}") from repair_error



def validate_records(raw: dict[str, Any], batch: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise ValueError("Gemini response schema invalid: expected object with results array")
    batch_by_id = {x["response_turn"]["turn_id"]: x for x in batch}
    clean, seen = [], set()
    returned = {}
    for item in raw["results"]:
        if not isinstance(item, dict):
            continue
        rid = item.get("response_turn_id")
        if rid not in batch_by_id or rid in returned:
            continue
        returned[rid] = item
    # Guarantee one retained record for every eligible response turn.
    for rid, case in batch_by_id.items():
        item = returned.get(rid, {})
        response = case["response_turn"]
        prior = {p["turn_id"]: p for p in case["prior_turns"]}
        model_classification = item.get("classification")
        classification = model_classification if model_classification in LABELS else "uncertain"
        target_id = item.get("target_turn_id")
        target = prior.get(target_id) if target_id else None
        if target is None:
            target_id = None
            classification = "uncertain"
        evidence_obj = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        statement_1 = evidence_obj.get("statement_1", "")
        statement_2 = evidence_obj.get("statement_2", "")
        statement_1 = statement_1 if isinstance(statement_1, str) else ""
        statement_2 = statement_2 if isinstance(statement_2, str) else ""
        target_quote_ok = bool(target and statement_1 and statement_1 in target["text"])
        response_quote_ok = bool(statement_2 and statement_2 in response["text"])
        evidence_verified = target_quote_ok and response_quote_ok
        proposed_decisive = model_classification in {"agreement", "partial_agreement", "disagreement"}
        if proposed_decisive and not evidence_verified:
            classification = "uncertain"
        immediate = bool(target and target["turn_id"] == case.get("immediate_previous_turn_id"))
        non_immediate = bool(target and not immediate)
        marker = "*" if non_immediate else ""
        key = (rid, target_id)
        if key in seen:
            continue
        seen.add(key)
        clean.append({
            "video_id": video_id, "interaction_id": f"INT_{len(clean)+1:04d}",
            "timestamp": round(float(response["start"]), 3),
            "response_turn_id": rid, "target_turn_id": target_id,
            "subtopic_id": None,
            "source_speaker": target["speaker_id"] if target else None,
            "target_speaker": target["speaker_id"] if target else None,
            "responding_speaker": response["speaker_id"],
            "classification": classification,
            "non_immediate_target": non_immediate, "marker": marker,
            "evidence": {"statement_1": statement_1 if target_quote_ok else "",
                         "statement_2": statement_2 if response_quote_ok else ""},
            "evidence_present": bool(statement_1 or statement_2),
            "evidence_found_in_target": target_quote_ok,
            "evidence_found_in_response": response_quote_ok,
            "evidence_validation_status": "verified" if evidence_verified else
                ("missing" if not (statement_1 or statement_2) else "invalid"),
            "manual_review_required": bool(proposed_decisive and not evidence_verified),
            "reasoning": normalize_text(item.get("reasoning"))[:1000],
            "confidence": item.get("confidence") if item.get("confidence") in {"high", "medium", "low"} else "low",
            "review_status": "manual_review_required" if proposed_decisive and not evidence_verified else "pending",
            "source_turn_timestamps": {
                "target_start": round(float(target["start"]), 3) if target else None,
                "target_end": round(float(target["end"]), 3) if target else None,
                "response_start": round(float(response["start"]), 3),
                "response_end": round(float(response["end"]), 3)}})
    return clean



def load_host_speaker(root: Path, video_id: str, explicit_path: Path | None = None):
    base = root / "data/v2/debate_only/videos" / video_id / VARIANT / "host_identification" / "host_identification.json"
    path = explicit_path if explicit_path else base
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return None, None, None, {"status": "missing", "warning": f"Host identification file not found: {path}"}
    try:
        data = read_json(path)
        if data.get("video_id") not in (None, video_id):
            raise ValueError(f"video_id mismatch: {data.get('video_id')}")
        speaker = data.get("predicted_host_speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("predicted_host_speaker is missing or invalid")
        return speaker.strip(), str(path), data, {"status": "loaded", "warning": data.get("warning")}
    except Exception as exc:
        return None, str(path), None, {"status": "invalid", "warning": str(exc)}

def host_interaction_summary(interactions: list[dict[str, Any]], host_speaker: str | None) -> dict[str, Any] | None:
    if not host_speaker:
        return None
    host_initiated = [x for x in interactions if x.get("responding_speaker") == host_speaker]
    host_received = [x for x in interactions if x.get("source_speaker") == host_speaker]
    guest_to_guest = [
        x for x in interactions
        if x.get("responding_speaker") != host_speaker
        and x.get("source_speaker") != host_speaker
        and x.get("source_speaker") is not None
    ]
    def counts(events):
        c = Counter(x["classification"] for x in events)
        return {label: c.get(label, 0) for label in
                ("agreement", "partial_agreement", "disagreement", "uncertain")}
    return {
        "host_speaker_id": host_speaker,
        "host_initiated_interactions": len(host_initiated),
        "host_initiated_classification_counts": counts(host_initiated),
        "host_received_interactions": len(host_received),
        "host_received_classification_counts": counts(host_received),
        "guest_to_guest_interactions": len(guest_to_guest),
        "guest_to_guest_classification_counts": counts(guest_to_guest),
        "host_involved_interactions": len(set(x["interaction_id"] for x in host_initiated + host_received)),
    }

def aggregate(interactions, turns, subtopics, host_speaker=None):
    labels = ["agreement","partial_agreement","disagreement","uncertain"]
    overall = Counter(x["classification"] for x in interactions)
    directional = [x for x in interactions if x.get("source_speaker") is not None]
    excluded_null_source = len(interactions) - len(directional)
    speakers = sorted({t["speaker_id"] for t in turns})
    by_speaker = {}
    for speaker in speakers:
        initiated = [x for x in directional if x.get("source_speaker") == speaker]
        received = [x for x in interactions if x.get("responding_speaker") == speaker]
        by_speaker[speaker] = {
            "initiated": dict(Counter(x["classification"] for x in initiated)),
            "received": dict(Counter(x["classification"] for x in received)),
            "total_initiated": len(initiated), "total_received": len(received)
        }
    pairs = Counter((x["source_speaker"],x["responding_speaker"],x["classification"]) for x in directional)
    pairwise = [{"source_speaker":a,"responding_speaker":b,"classification":c,"count":n}
                for (a,b,c),n in sorted(pairs.items(),key=lambda kv:(-kv[1],kv[0][0],kv[0][1] or "",kv[0][2]))]
    by_subtopic=[]
    for st in subtopics:
        events=[x for x in interactions if x.get("subtopic_id")==st.get("subtopic_id")]
        counts=Counter(x["classification"] for x in events)
        by_subtopic.append({"subtopic_id":st.get("subtopic_id"),"title":st.get("title"),
                            "start":st.get("start"),"end":st.get("end"),
                            "interaction_count":len(events),
                            "classification_counts":{label:counts.get(label,0) for label in labels}})
    decisive=sum(overall.get(x,0) for x in ("agreement","partial_agreement","disagreement"))
    agree=overall.get("agreement",0)+overall.get("partial_agreement",0)
    disagree=overall.get("disagreement",0)
    return {
        "classification_counts":{label:overall.get(label,0) for label in labels},
        "total_interactions":len(interactions),
        "decisive_interaction_count":decisive,"agreement_family_count":agree,
        "disagreement_family_count":disagree,
        "agreement_to_disagreement_ratio":round(agree/disagree,4) if disagree else None,
        "agreement_family_share_percent":round(agree/decisive*100,2) if decisive else None,
        "disagreement_family_share_percent":round(disagree/decisive*100,2) if decisive else None,
        "speaker_statistics":by_speaker,"directional_speaker_pairs":pairwise,
        "subtopic_statistics":by_subtopic,
        "host_statistics":host_interaction_summary(interactions,host_speaker),
        "directional_aggregation_exclusions":{"null_source_speaker_interactions":excluded_null_source,
                                               "included_in_directional_pairs":len(directional)},
        "ratio_note":"Agreement family = agreement + partial_agreement; disagreement family = disagreement. Uncertain cases are excluded from ratio denominators."
    }

def verify_language(merge):
    pipeline_language = ((merge.get("pipeline") or {}).get("asr") or {}).get("language")
    episode_language = (merge.get("episode") or {}).get("language")
    def norm(value):
        if value is None: return None
        value = str(value).strip().lower()
        if value in {"en","eng","english"}: return "English"
        return value
    p, e = norm(pipeline_language), norm(episode_language)
    warnings = []
    if p is None and e is None:
        return "unknown", {"status":"unverified","pipeline_asr_language":None,"episode_language":None,
                           "warnings":["Language metadata is missing from pipeline.asr.language and episode.language."]}
    if p and e and p != e:
        warnings.append(f"Language metadata inconsistent: pipeline.asr.language={p}, episode.language={e}.")
    chosen = p or e
    if chosen is None:
        return "unknown", {"status":"unverified","pipeline_asr_language":p,
                           "episode_language":e,"warnings":["Language metadata is unavailable."]}
    return str(chosen), {"status":"verified_from_source_metadata","pipeline_asr_language":p,
                         "episode_language":e,"warnings":warnings}


def validate_final_output(interactions, turns, subtopics):
    turn_map={t["turn_id"]:t for t in turns}
    speaker_ids={t["speaker_id"] for t in turns}
    subtopic_ids={s.get("subtopic_id") for s in subtopics}
    seen,issues=set(),[]
    for x in interactions:
        iid=x.get("interaction_id")
        if iid in seen: issues.append(f"duplicate_interaction_id:{iid}")
        seen.add(iid)
        if x.get("classification") not in LABELS: issues.append(f"invalid_interaction_classification:{iid}")
        response=turn_map.get(x.get("response_turn_id"))
        target=turn_map.get(x.get("target_turn_id")) if x.get("target_turn_id") else None
        if response is None: issues.append(f"unknown_response_turn:{iid}")
        if x.get("target_turn_id") is not None and target is None: issues.append(f"unknown_target_turn:{iid}")
        if response and x.get("responding_speaker") != response.get("speaker_id"):
            issues.append(f"responding_speaker_turn_mismatch:{iid}")
        if target and x.get("source_speaker") != target.get("speaker_id"):
            issues.append(f"source_speaker_turn_mismatch:{iid}")
        if x.get("responding_speaker") not in speaker_ids: issues.append(f"unknown_responding_speaker:{iid}")
        if x.get("source_speaker") is not None and x.get("source_speaker") not in speaker_ids:
            issues.append(f"unknown_source_speaker:{iid}")
        if x.get("subtopic_id") is not None and x.get("subtopic_id") not in subtopic_ids:
            issues.append(f"unknown_subtopic:{iid}")
        if response:
            expected=assign_subtopic(float(response["start"]),subtopics)
            expected_id=expected.get("subtopic_id") if expected else None
            if x.get("subtopic_id") != expected_id: issues.append(f"response_timestamp_subtopic_mismatch:{iid}")
        ts=x.get("source_turn_timestamps") or {}
        if ts.get("response_start") is None or ts.get("response_end") is None or ts["response_end"] < ts["response_start"]:
            issues.append(f"invalid_response_timestamps:{iid}")
        if x.get("evidence_found_in_response") and not x.get("evidence_present"):
            issues.append(f"evidence_validation_inconsistent:{iid}")
        if x.get("classification") in {"agreement","partial_agreement","disagreement"} and not (x.get("evidence_found_in_target") and x.get("evidence_found_in_response")):
            issues.append(f"decisive_without_verified_evidence:{iid}")
    directional=[x for x in interactions if x.get("source_speaker") is not None]
    if sum(1 for x in interactions if x.get("source_speaker") is None)+len(directional)!=len(interactions):
        issues.append("directional_aggregation_input_count_mismatch")
    return {"status":"passed" if not issues else "failed","checks_run":[
        "classification_labels","speaker_ids","target_response_turn_references","evidence_validation",
        "subtopic_mapping","directional_aggregation","missing_source_speakers",
        "duplicate_interaction_ids","timestamp_consistency","model_response_schema"],
        "issue_count":len(issues),"issues":issues}


def analyze_video(root: Path, video_id: str, model: str, api_key: str, resume: bool = True,
                  host_file: Path | None = None, force: bool = False) -> Path:
    base = root / "data/v2/debate_only"
    transcript_path = base / "videos" / video_id / VARIANT / "merge_final.json"
    subtopic_path = base / "subtopic_analysis" / VARIANT / video_id / "subtopic_analysis.json"
    overall_path = base / "overall_topic" / VARIANT / video_id / "overall_topic.json"
    host_default = base / "videos" / video_id / VARIANT / "host_identification" / "host_identification.json"
    out_dir = base / "agreement_disagreement" / VARIANT / video_id
    out_path = out_dir / "agreement_disagreement.json"
    checkpoint_path = out_dir / ".agreement_disagreement_checkpoint.json"
    if not transcript_path.exists(): raise FileNotFoundError(f"Missing transcript: {transcript_path}")
    if not subtopic_path.exists(): raise FileNotFoundError(f"Missing subtopic analysis: {subtopic_path}")
    merge, subtopic_data = read_json(transcript_path), read_json(subtopic_path)
    if subtopic_data.get("variant") not in (None, VARIANT): raise ValueError("Subtopic variant mismatch")
    if subtopic_data.get("video_id") not in (None, video_id): raise ValueError("Subtopic video_id mismatch")
    subtopics = subtopic_data.get("subtopics") or []
    if not subtopics: raise ValueError("No subtopics found")
    overall = read_json(overall_path) if overall_path.exists() else {}
    host_speaker, host_path, host_data, host_status = load_host_speaker(root, video_id, host_file or host_default)
    turns = build_turns(merge)
    speaker_ids = {t["speaker_id"] for t in turns}
    if host_speaker and host_speaker not in speaker_ids:
        host_status = {"status":"invalid_speaker_reference","warning":f"Host speaker {host_speaker} is not present in reconstructed transcript speakers."}
        host_speaker = None
    language, language_status = verify_language(merge)
    batches = build_target_batches(turns, subtopics)
    out_dir.mkdir(parents=True, exist_ok=True)
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()
    completed = {}
    checkpoint_meta = {"script_version":SCRIPT_VERSION,"schema_version":SCHEMA_VERSION,
                       "model":model,"prompt_version":PROMPT_VERSION,
                       "transcript_reconstruction_config":RECONSTRUCTION_CONFIG}
    if resume and checkpoint_path.exists():
        try:
            cp=read_json(checkpoint_path)
            compatible = all(cp.get(k)==v for k,v in checkpoint_meta.items()) and cp.get("video_id")==video_id and cp.get("variant")==VARIANT
            if compatible:
                completed={int(k):v for k,v in cp.get("completed_batches",{}).items()}
                print(f"  Resuming compatible v2.0 checkpoint: {len(completed)}/{len(batches)} batches")
            else:
                print("  Existing checkpoint metadata is incompatible; starting fresh.")
        except Exception as exc:
            print(f"  Warning: ignoring unreadable checkpoint: {exc}")
    all_interactions=[]
    for bi,batch in enumerate(batches):
        if bi in completed:
            records=completed[bi]
        else:
            payload={"video_id":video_id,"variant":VARIANT,"episode_topic":overall.get("overall_topic",{}),
                     "instructions":"Classify every response turn using full wording and prior context. Apply moderator/anchor distinctions in the system prompt. Host identity is metadata only.",
                     "response_turn_cases":[{"response_turn":x["response_turn"],"previous_context_turns":x["prior_turns"],
                                             "subtopic_context":x["subtopic"]} for x in batch]}
            print(f"  Gemini batch {bi+1}/{len(batches)} ({len(batch)} response turns)",flush=True)
            raw_path = out_dir / "raw_gemini_responses" / f"batch_{bi+1:03d}.txt"
            try:
                raw = call_gemini(api_key, model, payload, raw_dump_path=raw_path)
            except ValueError as batch_error:
                if len(batch) <= 1:
                    raise
                print(f"  Batch JSON recovery failed; splitting {len(batch)} turns into smaller calls", flush=True)
                repaired_results = []
                midpoint = (len(batch) + 1) // 2
                for part_no, part in enumerate((batch[:midpoint], batch[midpoint:]), 1):
                    if not part:
                        continue
                    part_payload = dict(payload)
                    part_payload["response_turn_cases"] = [
                        {"response_turn": x["response_turn"],
                         "previous_context_turns": x["prior_turns"],
                         "subtopic_context": x["subtopic"]} for x in part]
                    part_path = out_dir / "raw_gemini_responses" / f"batch_{bi+1:03d}_part_{part_no}.txt"
                    part_raw = call_gemini(api_key, model, part_payload, raw_dump_path=part_path)
                    repaired_results.extend(part_raw["results"])
                raw = {"results": repaired_results}
            if not isinstance(raw,dict) or not isinstance(raw.get("results"),list):
                raise ValueError(f"Gemini response schema invalid in batch {bi+1}")
            records=validate_records(raw,batch,video_id)
            for rec in records:
                st=assign_subtopic(rec["timestamp"],subtopics)
                rec["subtopic_id"]=st.get("subtopic_id") if st else None
            completed[bi]=records
            cp={**checkpoint_meta,"video_id":video_id,"variant":VARIANT,
                "completed_batches":{str(k):v for k,v in completed.items()},
                                "updated_at_utc":datetime.now(timezone.utc).isoformat()}
            checkpoint_path.write_text(json.dumps(cp,ensure_ascii=False,indent=2),encoding="utf-8")
        all_interactions.extend(records)
    all_interactions.sort(key=lambda x:(x["timestamp"],x["responding_speaker"],x.get("target_turn_id") or ""))
    for i,event in enumerate(all_interactions,1): event["interaction_id"]=f"INT_{i:04d}"
    summary=aggregate(all_interactions,turns,subtopics,host_speaker)
    validation=validate_final_output(all_interactions,turns,subtopics)
    evidence_summary={
        "records_with_evidence":sum(bool(x["evidence_present"]) for x in all_interactions),
        "records_with_verified_evidence":sum(x["evidence_validation_status"]=="verified" for x in all_interactions),
        "records_with_missing_evidence":sum(x["evidence_validation_status"]=="missing" for x in all_interactions),
        "records_with_invalid_evidence":sum(x["evidence_validation_status"]=="invalid" for x in all_interactions),
        "decisive_records_requiring_manual_review":sum(bool(x["manual_review_required"]) for x in all_interactions)}
    warnings=list(language_status["warnings"])
    if host_status.get("warning"): warnings.append(f"Host identification: {host_status['warning']}")
    if validation["status"]!="passed": warnings.append("One or more automated consistency checks failed.")
    if evidence_summary["decisive_records_requiring_manual_review"]:
        warnings.append("One or more model-proposed decisive interactions lacked verified evidence and were downgraded to uncertain.")
    duration=float((merge.get("episode") or {}).get("duration_s") or max(t["end"] for t in turns))
    output={
      "schema_version":SCHEMA_VERSION,"script_version":SCRIPT_VERSION,
      "metadata":{"video_id":video_id,"variant":VARIANT,"title":(merge.get("episode") or {}).get("title"),
                  "language":language,"language_verification":language_status,"model":model,
                  "host_speaker_id":host_speaker,"host_identification_status":host_status,
                  "host_prediction_confidence_band":(host_data or {}).get("confidence_band"),
                  "host_prediction_warning":(host_data or {}).get("warning"),
                  "generated_at_utc":datetime.now(timezone.utc).isoformat(),
                  "source_files":{"transcript":str(transcript_path),"subtopics":str(subtopic_path),
                                  "overall_topic":str(overall_path) if overall_path.exists() else None,
                                  "host_identification":host_path}},
      "methodology":{"analysis_unit":"directed speaker-to-speaker stance interaction",
                     "transcript_scope":"full transcript; not limited to overlapping speech",
                     "context_policy":f"Each response turn is assessed against up to {CONTEXT_TURNS} immediately preceding chronological turns; target may be any different speaker in that window.",
                     "classification_labels":sorted(LABELS),
                     "agreement_family":["agreement","partial_agreement"],
                     "disagreement_family":["disagreement"],
                     "moderator_policy":"Moderation, introductions, summaries, follow-up questions, and repetition without a stance are uncertain; substantive stances are classified normally. Host identity is not used for classification.",
                     "overlap_policy":"Overlap and interruption are not evidence of stance by themselves.",
                     "uncertain_policy":"Every eligible response turn receives one record; unclear or non-stance turns are classified uncertain.",
                     "target_policy":"Target is a specific earlier turn; no reverse inference. Response timestamp determines subtopic mapping.",
                     "limitations":["Automated classifications require manual validation.","ASR errors can affect stance and exact quote matching.","The analysis does not infer intent, ideology, bias, or factual correctness."]},
      "episode_summary":{"duration_seconds":round(duration,3),"speaker_count":len(speaker_ids),
                         "transcript_turn_count":len(turns),"subtopic_count":len(subtopics),
                         "candidate_response_turn_count":sum(len(b) for b in batches),
                         "candidate_classification_counts":summary["classification_counts"],**summary},
      "interactions":all_interactions,
      "data_quality":{"turns_with_text":len(turns),"turns_without_text_skipped":TURN_BUILD_DIAGNOSTICS["turns_without_text_skipped"],
                      "turn_reconstruction":TURN_BUILD_DIAGNOSTICS,
                      "host_identification_loaded":bool(host_speaker),"host_identification_status":host_status,
                      "subtopics_loaded":len(subtopics),
                      "interactions_missing_target":sum(not x.get("target_turn_id") for x in all_interactions),
                      "evidence_validation_summary":evidence_summary,
                      "validation":validation,"validation_warnings":warnings,
                      "manual_review_status":"required" if (evidence_summary["decisive_records_requiring_manual_review"] or validation["status"]!="passed" or warnings) else "pending_manual_validation",
                      "review_status":"pending_manual_validation"}
    }
    out_path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    if checkpoint_path.exists(): checkpoint_path.unlink()
    return out_path

def discover_videos(root: Path) -> list[str]:
    videos_dir = root / "data/v2/debate_only/videos"
    return sorted(p.name for p in videos_dir.iterdir() if (p / VARIANT / "merge_final.json").exists()) if videos_dir.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Agreement/disagreement analysis v2.0 (threshold_5.0 only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Process one video")
    group.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    parser.add_argument("--variant", choices=[VARIANT], default=VARIANT, help="Only threshold_5.0 is supported")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--force", action="store_true", help="Delete existing checkpoint and restart all batches")
    parser.add_argument("--host-file", type=Path, help="Optional host-identification JSON for --video-id; auto-discovery is also attempted")
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.", file=sys.stderr)
        return 2
    root = args.root.resolve()
    if args.all:
        video_ids = discover_videos(root)
    else:
        video_ids = [args.video_id]

    if not video_ids:
        print(f"No videos found under data/v2/debate_only/videos/*/{VARIANT}/merge_final.json")
        return 1    
    
    processed = skipped = failed = 0
    for video_id in video_ids:
        print(f"\n{'=' * 65}\nAgreement/disagreement: {video_id} ({VARIANT})\n{'=' * 65}", flush=True)
        try:
            path = analyze_video(root, video_id, args.model, api_key, resume=not args.no_resume, force=args.force,
                                 host_file=args.host_file if video_id == args.video_id else None)
            print(f"SUCCESS: {path}", flush=True)
            processed += 1
        except FileNotFoundError as exc:
            print(f"SKIPPED: {exc}", file=sys.stderr)
            skipped += 1
        except Exception as exc:
            print(f"FAILED: {video_id}: {exc}", file=sys.stderr)
            failed += 1
    print(f"\nBATCH SUMMARY: processed={processed}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())