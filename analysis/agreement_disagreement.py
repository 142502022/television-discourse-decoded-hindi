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
SCRIPT_VERSION = "word-level-host-aware-1.1"
SCHEMA_VERSION = "agreement-disagreement-1.2"
PROMPT_VERSION = "moderation-aware-1.2"
RECONSTRUCTION_CONFIG = {"method": "word_midpoint_to_closest_same_speaker_interval", "fallback_gap_seconds": 1.5}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
LABELS = {"agreement", "partial_agreement", "disagreement", "counterargument", "neutral", "uncertain"}
BATCH_TARGETS = 5
CONTEXT_TURNS = 5

SYSTEM_PROMPT = r"""You are analyzing an ENGLISH televised debate transcript.
Identify DIRECTED STANCE INTERACTIONS between different speakers only.

Allowed labels:
- agreement: clearly endorses/supports the target speaker's claim.
- partial_agreement: accepts a meaningful part but qualifies/rejects another part.
- disagreement: explicitly rejects or contradicts the target claim.
- counterargument: gives a substantive opposing reason, fact, or explanation.
- neutral: no discernible stance toward an earlier speaker's claim.
- uncertain: evidence/context is insufficient or the target cannot be identified reliably.

IMPORTANT MODERATOR / ANCHOR DISTINCTION:
Assess the actual words and context, not the speaker's presumed role or host identity.
A turn may be neutral when it merely:
- moderates the discussion or manages the floor;
- introduces a panelist or directs a question to them;
- summarizes or restates another speaker's position without endorsing it;
- asks a follow-up or clarification question without taking a side;
- repeats a claim without expressing support or opposition.
These are not automatically neutral: if the speaker adds a substantive stance, classify that stance normally.
A substantive opposing reason is counterargument; explicit rejection is disagreement;
clear support is agreement or partial_agreement as appropriate.

Rules:
1. Do NOT classify from keywords alone. Analyze contextual meaning, not sentiment.
2. A question is neutral unless it clearly expresses a stance; challenge-style wording alone is insufficient.
3. Identify the specific earlier turn being addressed. Use target_turn_id only when supported.
4. Do not assume the immediately previous speaker is the target; never infer the reverse direction.
5. Create an interaction only when the response turn contains a discernible stance toward the target claim.
6. Evidence must be an exact, contiguous quotation copied from the responding turn, under 35 words.
   Do not normalize, paraphrase, repair, or fabricate the quotation.
7. If the classification is decisive (agreement, partial_agreement, disagreement,
   counterargument), it MUST have non-empty exact evidence. If no exact evidence exists,
   return uncertain and leave evidence empty.
8. Keep neutral turns out of interaction records. Still return one neutral result for each
   supplied response turn so candidate/classification counts can be audited.
9. Uncertain cases may have empty evidence and must remain uncertain.
10. Host identity is metadata only and MUST NOT affect classification.
11. Self-continuation is not a directed interaction. Overlap or interruption alone is not stance.
12. Avoid duplicate records for the same response/target pair.

Return ONLY valid JSON:
{"results":[{"response_turn_id":"T0002","target_turn_id":"T0001",
"classification":"agreement|partial_agreement|disagreement|counterargument|neutral|uncertain",
"evidence":"exact quote from response turn","reasoning":"brief contextual explanation",
"confidence":"high|medium|low"}]}
Return one result per supplied response turn. If no interaction exists, return neutral,
target_turn_id null, and evidence empty. Never output markdown fences."""


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
            "subtopic": ({"subtopic_id": st.get("subtopic_id"), "title": st.get("title"),
                          "description": st.get("description")} if st else None),
        })
    return [targets[i:i + BATCH_TARGETS] for i in range(0, len(targets), BATCH_TARGETS)]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Gemini response did not contain valid JSON")
        data = json.loads(match.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Gemini response must contain a results array")
    return data


def call_gemini(api_key: str, model: str, payload: dict[str, Any], max_retries: int = 5) -> dict[str, Any]:
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        API_URL.format(model=model, key=api_key),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = "".join(part.get("text", "") for candidate in result.get("candidates", [])
                           for part in candidate.get("content", {}).get("parts", []))
            if not text:
                raise ValueError(f"Gemini returned no text: {json.dumps(result)[:1000]}")
            return extract_json(text)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries - 1:
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


def validate_records(raw: dict[str, Any], batch: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise ValueError("Gemini response schema invalid: expected object with results array")
    allowed = {x["response_turn"]["turn_id"] for x in batch}
    turn_map = {x["response_turn"]["turn_id"]: x["response_turn"] for x in batch}
    prior_map = {x["response_turn"]["turn_id"]: {p["turn_id"]: p for p in x["prior_turns"]} for x in batch}
    clean, seen = [], set()
    for item in raw["results"]:
        if not isinstance(item, dict):
            continue
        rid = item.get("response_turn_id")
        if rid not in allowed:
            continue
        model_classification = item.get("classification")
        classification = model_classification
        if classification not in LABELS:
            classification = "uncertain"
            model_classification = "invalid_label"
        target_id = item.get("target_turn_id")
        if target_id not in prior_map[rid]:
            target_id = None
            if classification not in {"neutral", "uncertain"}:
                classification = "uncertain"
        response = turn_map[rid]
        target = prior_map[rid].get(target_id) if target_id else None
        if classification == "neutral":
            continue
        if classification != "uncertain" and target is None:
            classification = "uncertain"
        evidence_raw = item.get("evidence")
        evidence = evidence_raw if isinstance(evidence_raw, str) else ""
        evidence_present = bool(evidence.strip())
        # Exact contiguous substring only: no whitespace normalization or case-folding.
        evidence_found = bool(evidence_present and evidence in response["text"])
        proposed_decisive = model_classification in {"agreement","partial_agreement","disagreement","counterargument"}
        if not evidence_found and proposed_decisive:
            classification = "uncertain"
        status = "verified" if evidence_found else ("missing" if not evidence_present else "invalid")
        decisive = proposed_decisive
        key = (rid, target_id, classification)
        if key in seen:
            continue
        seen.add(key)
        clean.append({
            "video_id": video_id, "interaction_id": f"INT_{len(clean)+1:04d}",
            "timestamp": round(float(response["start"]),3), "response_turn_id": rid,
            "target_turn_id": target_id, "subtopic_id": None,
            "source_speaker": target["speaker_id"] if target else None,
            "responding_speaker": response["speaker_id"], "classification": classification,
            "evidence": evidence, "evidence_present": evidence_present,
            "evidence_found_in_response": evidence_found, "evidence_validation_status": status,
            "manual_review_required": bool(decisive and not evidence_found),
            "reasoning": normalize_text(item.get("reasoning"))[:1000],
            "confidence": item.get("confidence") if item.get("confidence") in {"high","medium","low"} else "low",
            "review_status": "manual_review_required" if decisive and not evidence_found else "pending",
            "source_turn_timestamps": {
                "target_start": round(float(target["start"]),3) if target else None,
                "target_end": round(float(target["end"]),3) if target else None,
                "response_start": round(float(response["start"]),3),
                "response_end": round(float(response["end"]),3)
            }
        })
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
                ("agreement", "partial_agreement", "disagreement", "counterargument", "uncertain")}
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
    labels = ["agreement","partial_agreement","disagreement","counterargument","uncertain"]
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
    decisive=sum(overall.get(x,0) for x in ("agreement","partial_agreement","disagreement","counterargument"))
    agree=overall.get("agreement",0)+overall.get("partial_agreement",0)
    disagree=overall.get("disagreement",0)+overall.get("counterargument",0)
    return {
        "classification_counts":{label:overall.get(label,0) for label in labels},
        "total_non_neutral_interactions":len(interactions),
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
        "ratio_note":"Agreement family = agreement + partial_agreement; disagreement family = disagreement + counterargument. Neutral turns are omitted; uncertain cases are excluded from ratio denominators."
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
    if chosen != "English":
        warnings.append(f"Source metadata does not explicitly identify English (resolved value: {chosen}).")
        return "unknown", {"status":"inconsistent_or_non_english","pipeline_asr_language":p,
                           "episode_language":e,"warnings":warnings}
    return "English", {"status":"verified_from_source_metadata","pipeline_asr_language":p,
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
        if x.get("classification") not in LABELS-{"neutral"}: issues.append(f"invalid_interaction_classification:{iid}")
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
        if x.get("classification") in {"agreement","partial_agreement","disagreement","counterargument"} and not x.get("evidence_found_in_response"):
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
    completed, neutral_counts = {}, {}
    checkpoint_meta = {"script_version":SCRIPT_VERSION,"schema_version":SCHEMA_VERSION,
                       "model":model,"prompt_version":PROMPT_VERSION,
                       "transcript_reconstruction_config":RECONSTRUCTION_CONFIG}
    if resume and checkpoint_path.exists():
        try:
            cp=read_json(checkpoint_path)
            compatible = all(cp.get(k)==v for k,v in checkpoint_meta.items()) and cp.get("video_id")==video_id and cp.get("variant")==VARIANT
            if compatible:
                completed={int(k):v for k,v in cp.get("completed_batches",{}).items()}
                neutral_counts={int(k):int(v) for k,v in cp.get("neutral_candidate_counts",{}).items()}
                print(f"  Resuming compatible v1.2 checkpoint: {len(completed)}/{len(batches)} batches")
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
            raw=call_gemini(api_key,model,payload)
            if not isinstance(raw,dict) or not isinstance(raw.get("results"),list):
                raise ValueError(f"Gemini response schema invalid in batch {bi+1}")
            neutral_counts[bi]=sum(1 for r in raw["results"] if isinstance(r,dict) and r.get("classification")=="neutral")
            records=validate_records(raw,batch,video_id)
            for rec in records:
                st=assign_subtopic(rec["timestamp"],subtopics)
                rec["subtopic_id"]=st.get("subtopic_id") if st else None
            completed[bi]=records
            cp={**checkpoint_meta,"video_id":video_id,"variant":VARIANT,
                "completed_batches":{str(k):v for k,v in completed.items()},
                "neutral_candidate_counts":{str(k):v for k,v in neutral_counts.items()},
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
                     "context_policy":f"Each response turn is assessed with up to {CONTEXT_TURNS} preceding different-speaker turns.",
                     "classification_labels":sorted(LABELS),
                     "agreement_family":["agreement","partial_agreement"],
                     "disagreement_family":["disagreement","counterargument"],
                     "moderator_policy":"Moderation, panelist introductions, summaries, follow-up questions, and repetition without a stance are neutral; substantive anchor stances are classified normally. Host identity is not used for classification.",
                     "overlap_policy":"Overlap and interruption are not evidence of stance by themselves.",
                     "neutral_policy":"Neutral candidates are counted in the candidate summary but omitted from interaction records; uncertain records are retained.",
                     "target_policy":"Target is a specific earlier turn; no reverse inference. Response timestamp determines subtopic mapping.",
                     "limitations":["Automated classifications require manual validation.","ASR errors can affect stance and exact quote matching.","The analysis does not infer intent, ideology, bias, or factual correctness."]},
      "episode_summary":{"duration_seconds":round(duration,3),"speaker_count":len(speaker_ids),
                         "transcript_turn_count":len(turns),"subtopic_count":len(subtopics),
                         "candidate_response_turn_count":sum(len(b) for b in batches),
                         "neutral_candidate_count":sum(neutral_counts.values()),
                         "candidate_classification_counts":{**summary["classification_counts"],
                             "neutral":sum(neutral_counts.values())},**summary},
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
    parser = argparse.ArgumentParser(description="English televised-debate agreement/disagreement analysis v1.2 (threshold_5.0 only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Process one video")
    group.add_argument("--all", action="store_true", help="Process all videos (not intended for v1.2 pilot)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    parser.add_argument("--variant", choices=[VARIANT], default=VARIANT, help="Only threshold_5.0 is supported")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--force", action="store_true", help="Delete any existing checkpoint and run fresh Gemini classification")
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