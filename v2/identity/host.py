"""Phase 2: interpretable host / moderator identification for one episode.

The host of a televised debate is inferred from *multiple* independent,
transparent signals. No single signal decides the result: speaking time,
greeting phrases, and interruption counts are deliberately capped so that a
candidate must also look like a moderator in other ways (asking questions,
addressing many distinct participants, speaking between participants, taking
back the floor after overlap).

Design rules:
- The role is stored separately from the diarization label. ``SPEAKER_xx`` is
  never renamed and is never assumed stable across episodes.
- A manually supplied reference label may be passed for *evaluation only*; it
  is never used as a rule that determines the prediction.
- Scores are heuristic and uncalibrated. They are reported as
  ``heuristic_score`` plus a coarse ``confidence_band``, never as probabilities.
- Phrases are discovered from the actual transcript; nothing is assumed to
  occur in a particular episode.

The module is pure and model-free so it can be unit-tested without audio.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

# ── lexicons ────────────────────────────────────────────────────────────────
# Interrogative openers. A segment starting with one of these is treated as a
# question even if ASR dropped the trailing "?".
QUESTION_STARTERS = {
    "what", "why", "how", "who", "whom", "whose", "which", "when", "where",
    "do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
    "would", "should", "shall", "has", "have", "had", "may", "might",
}

# Second-person forms that indicate the speaker is addressing someone.
ADDRESS_TOKENS = {"you", "your", "yours", "you're", "youre", "u", "sir",
                  "madam", "ma'am", "maam"}

# Greetings are reported as context only, never used as a primary signal.
GREETING_PHRASES = (
    "good morning", "good evening", "good afternoon", "good night",
    "welcome back", "welcome to", "welcome", "namaste", "hello", "hi everyone",
)

# Function words excluded when discovering repeated *content* phrases.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "up", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your", "yours",
    "yeah", "yes", "okay", "ok", "right", "just", "very", "much",
}

# Signals used to rank candidates, and their weights. Interrogative and
# conversational-role signals dominate; temporal signals are capped so that
# speaking time can never decide the result on its own.
SIGNAL_WEIGHTS: Dict[str, float] = {
    "question_share": 0.16,
    "distinct_interlocutors": 0.14,
    "turns_between_others": 0.12,
    "floor_regain": 0.10,
    "address_share": 0.10,
    "questions_per_minute": 0.10,
    "question_rate": 0.08,
    "repeated_phrases": 0.08,
    "speech_share": 0.06,
    "moderator_phrase_rate": 0.06,
}

# Rate-based signals are only trusted for speakers with enough data; otherwise
# a single question in a single segment would produce a spurious 100% rate.
MIN_SEGMENTS_FOR_RATE = 8
MIN_SECONDS_FOR_RATE = 30.0


# ── small numeric helpers ───────────────────────────────────────────────────
def _num(value: Any) -> Optional[float]:
    import math
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _r3(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _guarded_rate(numerator: float, denominator: float, enough: bool) -> float:
    """A ratio that is only trusted when the sample is large enough."""
    if not enough or denominator <= 0:
        return 0.0
    return numerator / denominator


# ── text helpers ────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, return word tokens (keeps digits)."""
    cleaned = []
    current = []
    for char in text.lower():
        if char.isalnum() or char == "'":
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []
    if current:
        cleaned.append("".join(current))
    return cleaned


def is_question(text: str) -> bool:
    """True if a segment reads as a question (explicit '?' or opener)."""
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    tokens = tokenize(stripped)
    return bool(tokens) and tokens[0] in QUESTION_STARTERS


def has_address(text: str) -> bool:
    """True if the segment contains a second-person address."""
    return bool(set(tokenize(text)) & ADDRESS_TOKENS)


def find_greetings(text: str) -> List[str]:
    lowered = text.lower()
    return [phrase for phrase in GREETING_PHRASES if phrase in lowered]


def repeated_phrases(
    texts: Sequence[str],
    min_count: int = 3,
    max_n: int = 4,
    top: int = 8,
) -> List[Dict[str, Any]]:
    """Discover repeated 2..max_n-word content phrases in a speaker's text.

    Phrases made entirely of stopwords are excluded so the output is
    interpretable. Ordering is deterministic: count desc, then phrase.
    """
    counts: Counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        for n in range(2, max_n + 1):
            for start in range(len(tokens) - n + 1):
                gram = tuple(tokens[start:start + n])
                if all(token in STOPWORDS for token in gram):
                    continue
                counts[gram] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    output = []
    for gram, count in ranked:
        if count < min_count:
            continue
        output.append({"phrase": " ".join(gram), "count": count})
        if len(output) >= top:
            break
    return output


# ── feature extraction ──────────────────────────────────────────────────────
def _iter_valid_turns(turns: Any) -> List[Dict[str, Any]]:
    result = []
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker")
        start, end = _num(turn.get("start")), _num(turn.get("end"))
        if not isinstance(speaker, str) or not speaker.strip():
            continue
        if start is None or end is None or end <= start:
            continue
        result.append({"speaker": speaker, "start": start, "end": end})
    return sorted(result, key=lambda t: (t["start"], t["end"], t["speaker"]))


def _iter_valid_segments(segments: Any) -> List[Dict[str, Any]]:
    result = []
    for index, segment in enumerate(segments if isinstance(segments, list) else []):
        if not isinstance(segment, dict):
            continue
        speaker = segment.get("speaker")
        start, end = _num(segment.get("start")), _num(segment.get("end"))
        text = segment.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            continue
        if not isinstance(text, str):
            continue
        result.append({"index": index, "speaker": speaker, "start": start,
                       "end": end, "text": text.strip()})
    return result


def _iter_valid_overlaps(overlaps: Any) -> List[Dict[str, Any]]:
    result = []
    for event in overlaps if isinstance(overlaps, list) else []:
        if not isinstance(event, dict):
            continue
        start, end = _num(event.get("start")), _num(event.get("end"))
        speakers = event.get("speakers")
        if start is None or end is None or end <= start:
            continue
        if not isinstance(speakers, list):
            speakers = []
        result.append({"start": start, "end": end,
                       "speakers": [s for s in speakers if isinstance(s, str)]})
    return result


def extract_features(
    segments: Any,
    turns: Any,
    overlaps: Any,
    *,
    trim_start_s: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Per-speaker features from diarized segments, turns and overlap events."""
    valid_turns = _iter_valid_turns(turns)
    valid_segments = _iter_valid_segments(segments)
    valid_overlaps = _iter_valid_overlaps(overlaps)

    features: Dict[str, Dict[str, Any]] = {}

    def ensure(speaker: str) -> Dict[str, Any]:
        if speaker not in features:
            features[speaker] = {
                "speech_seconds": 0.0,
                "turn_count": 0,
                "segment_count": 0,
                "question_count": 0,
                "address_count": 0,
                "distinct_interlocutors": set(),
                "turns_between_others": 0,
                "floor_regains": 0,
                "overlap_involvements": 0,
                "first_speech_s": None,
                "last_speech_s": None,
                "greeting_count": 0,
                "question_examples": [],
                "address_examples": [],
                "texts": [],
            }
        return features[speaker]

    # speaking time and turn-taking from diarization turns
    for turn in valid_turns:
        feat = ensure(turn["speaker"])
        feat["speech_seconds"] += turn["end"] - turn["start"]
        feat["turn_count"] += 1
        first = feat["first_speech_s"]
        feat["first_speech_s"] = turn["start"] if first is None else min(first, turn["start"])
        last = feat["last_speech_s"]
        feat["last_speech_s"] = turn["end"] if last is None else max(last, turn["end"])

    for index, turn in enumerate(valid_turns):
        speaker = turn["speaker"]
        prev = valid_turns[index - 1]["speaker"] if index > 0 else None
        nxt = valid_turns[index + 1]["speaker"] if index + 1 < len(valid_turns) else None
        if nxt is not None and nxt != speaker:
            ensure(speaker)["distinct_interlocutors"].add(nxt)
        if prev is not None and prev != speaker and nxt is not None and nxt != speaker and prev != nxt:
            ensure(speaker)["turns_between_others"] += 1

    # linguistic cues from segments
    for segment in valid_segments:
        speaker = segment["speaker"]
        feat = ensure(speaker)
        feat["segment_count"] += 1
        text = segment["text"]
        feat["texts"].append(text)
        if is_question(text):
            feat["question_count"] += 1
            if len(feat["question_examples"]) < 5:
                feat["question_examples"].append(
                    {"start": _r3(segment["start"]), "end": _r3(segment["end"]), "text": text[:200]})
        if has_address(text):
            feat["address_count"] += 1
            if len(feat["address_examples"]) < 5:
                feat["address_examples"].append(
                    {"start": _r3(segment["start"]), "end": _r3(segment["end"]), "text": text[:200]})
        feat["greeting_count"] += len(find_greetings(text))

    # floor regain after overlap
    for event in valid_overlaps:
        for speaker in event["speakers"]:
            ensure(speaker)["overlap_involvements"] += 1
            window_end = event["end"] + 2.0
            regained = any(
                turn["speaker"] == speaker and turn["start"] <= window_end and turn["end"] >= event["end"]
                for turn in valid_turns
            )
            if regained:
                ensure(speaker)["floor_regains"] += 1

    for speaker, feat in features.items():
        feat["distinct_interlocutor_count"] = len(feat["distinct_interlocutors"])
        feat["distinct_interlocutors"] = sorted(feat["distinct_interlocutors"])
        feat["speech_share"] = 0.0  # filled after total is known
        texts = feat.pop("texts")
        feat["repeated_phrases"] = repeated_phrases(texts)
        feat["repeated_phrase_count"] = len(feat["repeated_phrases"])

    total_speech = sum(feat["speech_seconds"] for feat in features.values())
    for feat in features.values():
        feat["speech_share"] = feat["speech_seconds"] / total_speech if total_speech else 0.0

    return features


# ── ranking ─────────────────────────────────────────────────────────────────
def percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
    """Map raw values to 0..1 percentile ranks (ties share the average rank)."""
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda pair: pair[1])
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 0.5}
    ranks: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        average_rank = (i + j) / 2.0
        percentile = average_rank / (n - 1)
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = percentile
        i = j + 1
    return ranks


def _raw_signal(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    signal_values: Dict[str, Dict[str, float]] = {name: {} for name in SIGNAL_WEIGHTS}
    total_questions = sum(feat["question_count"] for feat in features.values())
    total_addresses = sum(feat["address_count"] for feat in features.values())
    for speaker, feat in features.items():
        segments = feat["segment_count"]
        seconds = feat["speech_seconds"]
        minutes = seconds / 60.0
        signal_values["question_share"][speaker] = (
            feat["question_count"] / total_questions if total_questions else 0.0)
        signal_values["distinct_interlocutors"][speaker] = float(feat["distinct_interlocutor_count"])
        signal_values["turns_between_others"][speaker] = float(feat["turns_between_others"])
        signal_values["floor_regain"][speaker] = float(feat["floor_regains"])
        signal_values["address_share"][speaker] = (
            feat["address_count"] / total_addresses if total_addresses else 0.0)
        signal_values["questions_per_minute"][speaker] = _guarded_rate(
            feat["question_count"], minutes, seconds >= MIN_SECONDS_FOR_RATE)
        signal_values["question_rate"][speaker] = _guarded_rate(
            feat["question_count"], float(segments), segments >= MIN_SEGMENTS_FOR_RATE)
        signal_values["repeated_phrases"][speaker] = float(feat["repeated_phrase_count"])
        signal_values["speech_share"][speaker] = feat["speech_share"]
        signal_values["moderator_phrase_rate"][speaker] = _guarded_rate(
            float(feat["repeated_phrase_count"]), float(segments), segments >= MIN_SEGMENTS_FOR_RATE)
    return signal_values


def normalize_signals(
    signal_values: Dict[str, Dict[str, float]],
    *,
    compress: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Scale each signal to 0..1 by its maximum, then compress outliers.

    Dividing by the maximum keeps magnitude differences (a speaker who asks
    five times as many questions scores much higher), and the square-root
    compression (``compress=0.5``) stops one extreme value from drowning out
    every other signal.
    """
    normalized: Dict[str, Dict[str, float]] = {}
    for name, values in signal_values.items():
        top = max(values.values()) if values else 0.0
        if top <= 0:
            normalized[name] = {speaker: 0.0 for speaker in values}
            continue
        normalized[name] = {
            speaker: (value / top) ** compress for speaker, value in values.items()
        }
    return normalized


def rank_candidates(
    features: Dict[str, Dict[str, Any]],
    *,
    min_speech_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Rank speakers as host candidates using the weighted multi-signal score."""
    usable = {s: f for s, f in features.items() if f["speech_seconds"] >= min_speech_seconds}
    if not usable:
        usable = features
    signal_values = _raw_signal(usable)
    normalized = normalize_signals(signal_values)

    scores: Dict[str, float] = {speaker: 0.0 for speaker in usable}
    per_signal_winner: Dict[str, Optional[str]] = {}
    for name, weight in SIGNAL_WEIGHTS.items():
        values = signal_values[name]
        if not values:
            per_signal_winner[name] = None
            continue
        top_value = max(values.values())
        winners = [s for s, v in values.items() if v == top_value and v > 0]
        per_signal_winner[name] = sorted(winners)[0] if winners else None
        for speaker in usable:
            scores[speaker] += weight * normalized[name].get(speaker, 0.0)

    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return {
        "usable_speakers": sorted(usable),
        "scores": scores,
        "ranking": ordered,
        "signal_values": signal_values,
        "signal_normalized": normalized,
        "signal_percentiles": {name: percentile_ranks(values) for name, values in signal_values.items()},
        "signal_winner": per_signal_winner,
        "signal_weights": dict(SIGNAL_WEIGHTS),
    }


def _confidence_band(
    scores: Dict[str, float],
    ranking: List[tuple],
    signal_winner: Dict[str, Optional[str]],
    usable_count: int,
) -> Dict[str, Any]:
    if usable_count < 2 or not ranking:
        return {"band": "uncertain", "margin": None, "signal_agreement": None,
                "reason": "Fewer than two comparable speakers."}
    top_speaker, top_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0.0
    margin = top_score - second_score
    decisions = [winner for winner in signal_winner.values() if winner is not None]
    agreement = sum(1 for winner in decisions if winner == top_speaker) / len(decisions) if decisions else 0.0
    if margin >= 0.12 and agreement >= 0.5:
        band, reason = "high", "Large score margin and majority of signals agree."
    elif margin >= 0.05:
        band, reason = "medium", "Clear but not dominant margin over the runner-up."
    else:
        band, reason = "low", "Small margin over the runner-up; treat as uncertain."
    return {"band": band, "margin": _r3(margin), "signal_agreement": _r3(agreement),
            "reason": reason, "top_score": _r3(top_score), "second_score": _r3(second_score)}


def _evidence_block(speaker: str, feat: Dict[str, Any], signals: Dict[str, float]) -> Dict[str, Any]:
    segments = feat["segment_count"]
    minutes = feat["speech_seconds"] / 60.0
    return {
        "speaker_label": speaker,
        "speech_seconds": _r3(feat["speech_seconds"]),
        "speech_share": _r3(feat["speech_share"]),
        "turn_count": feat["turn_count"],
        "segment_count": segments,
        "question_count": feat["question_count"],
        "question_share": _r3(signals.get("question_share")),
        "question_rate": _r3(signals.get("question_rate")),
        "questions_per_minute": _r3(signals.get("questions_per_minute")),
        "address_count": feat["address_count"],
        "address_share": _r3(signals.get("address_share")),
        "distinct_interlocutor_count": feat["distinct_interlocutor_count"],
        "distinct_interlocutors": feat["distinct_interlocutors"],
        "turns_between_others": feat["turns_between_others"],
        "overlap_involvements": feat["overlap_involvements"],
        "floor_regains": feat["floor_regains"],
        "first_speech_s": _r3(feat["first_speech_s"]),
        "last_speech_s": _r3(feat["last_speech_s"]),
        "greeting_count": feat["greeting_count"],
        "repeated_phrases": feat["repeated_phrases"],
        "repeated_phrase_count": feat["repeated_phrase_count"],
        "moderator_phrase_rate": _r3(signals.get("moderator_phrase_rate")),
        "rate_sample_ok": segments >= MIN_SEGMENTS_FOR_RATE,
        "question_examples": feat["question_examples"],
        "address_examples": feat["address_examples"],
    }


def infer_host(
    segments: Any,
    turns: Any,
    overlaps: Any,
    *,
    reference_label: Optional[str] = None,
    trim_start_s: float = 0.0,
    top_n: int = 8,
) -> Dict[str, Any]:
    """Infer the host/moderator and return a fully interpretable result."""
    features = extract_features(segments, turns, overlaps, trim_start_s=trim_start_s)
    if not features:
        return {
            "prediction": {"speaker_label": None, "inferred_role": "unknown",
                           "confidence_band": "uncertain", "uncertain": True,
                           "reason": "No usable diarized speakers."},
            "reference_comparison": _reference_compare(None, reference_label, {}, {}),
            "candidates": [], "warnings": ["no_usable_speakers"],
        }

    ranked = rank_candidates(features, min_speech_seconds=5.0)
    ranking = ranked["ranking"]
    scores = ranked["scores"]
    usable = set(ranked["usable_speakers"])
    confidence = _confidence_band(scores, ranking, ranked["signal_winner"], len(usable))
    top_speaker = ranking[0][0] if ranking else None

    candidates = []
    for rank, (speaker, score) in enumerate(ranking[:top_n], 1):
        feat = features[speaker]
        signals = {name: ranked["signal_values"][name].get(speaker, 0.0) for name in SIGNAL_WEIGHTS}
        candidates.append({
            "rank": rank,
            "speaker_label": speaker,
            "inferred_role": "host_candidate" if rank == 1 else "participant_candidate",
            "heuristic_score": _r3(score),
            "evidence": _evidence_block(speaker, feat, signals),
        })

    # speakers that were too short to compete, listed separately (not hidden)
    short = sorted(s for s in features if s not in usable)
    warnings = []
    if confidence["band"] == "uncertain":
        warnings.append("insufficient_evidence_for_confident_host")
    if top_speaker is not None and features[top_speaker]["greeting_count"] == 0:
        warnings.append("no_greeting_detected_for_top_candidate")

    prediction = {
        "speaker_label": top_speaker,
        "inferred_role": "host_moderator" if top_speaker else "unknown",
        "confidence_band": confidence["band"],
        "uncertain": confidence["band"] in {"low", "uncertain"},
        "confidence": confidence,
        "supporting_signals": {
            name: {"weight": weight, "top_speaker_in_signal": ranked["signal_winner"][name]}
            for name, weight in SIGNAL_WEIGHTS.items()
        },
        "policy_note": "Scores are heuristic and uncalibrated; the role is not a person identity.",
    }

    return {
        "prediction": prediction,
        "reference_comparison": _reference_compare(top_speaker, reference_label, scores, ranked["signal_values"]),
        "candidates": candidates,
        "excluded_short_speakers": short,
        "signal_definitions": SIGNAL_DEFINITIONS,
        "warnings": warnings,
    }


def _reference_compare(
    predicted: Optional[str],
    reference: Optional[str],
    scores: Dict[str, float],
    signal_values: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    if not reference:
        return {"reference_label": None, "match": None,
                "note": "No reference label supplied; evaluation skipped."}
    if not scores:
        return {"reference_label": reference, "match": None,
                "note": "No scored candidates to compare against the reference."}
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    labels = [s for s, _ in ordered]
    rank = labels.index(reference) + 1 if reference in labels else None
    return {
        "reference_label": reference,
        "predicted_label": predicted,
        "match": bool(predicted == reference),
        "reference_rank": rank,
        "reference_heuristic_score": _r3(scores.get(reference)),
        "num_scored_candidates": len(ordered),
        "note": "Reference used for evaluation only; it did not influence ranking.",
    }


SIGNAL_DEFINITIONS: Dict[str, str] = {
    "question_share": "Share of all detected questions attributed to the speaker.",
    "distinct_interlocutors": "Number of different speakers who speak immediately after this speaker.",
    "turns_between_others": "Turns where this speaker is preceded and followed by two different other speakers.",
    "floor_regain": "Overlap events after which this speaker speaks again at/after the overlap ends.",
    "address_share": "Share of all second-person addresses attributed to the speaker.",
    "questions_per_minute": "Questions per minute of speech (only trusted above a minimum speaking time).",
    "question_rate": "Fraction of the speaker's segments that read as questions (guarded by a minimum segment count).",
    "repeated_phrases": "Count of repeated 2-4 word content phrases discovered in the speaker's text.",
    "speech_share": "Share of total diarized speaking time (weight-capped; never decisive alone).",
    "moderator_phrase_rate": "Repeated moderator-style phrases per segment (guarded by a minimum segment count).",
}


def build_result(
    final_json: Dict[str, Any],
    *,
    video_id: Optional[str] = None,
    reference_label: Optional[str] = None,
    trim_start_s: float = 0.0,
    top_n: int = 8,
) -> Dict[str, Any]:
    """Assemble the full host-identification artifact from a final.json dict."""
    episode = final_json.get("episode") if isinstance(final_json.get("episode"), dict) else {}
    participants = final_json.get("participants")
    result = infer_host(
        final_json.get("segments"),
        final_json.get("speaker_turns"),
        final_json.get("overlap_candidates"),
        reference_label=reference_label,
        trim_start_s=trim_start_s,
        top_n=top_n,
    )
    return {
        "schema_version": "host-identification-1.0",
        "method": "interpretable-multi-signal",
        "reference_policy": "A supplied reference label is used for evaluation only and never as a ranking rule.",
        "video": {
            "video_id": video_id or episode.get("video_id"),
            "title": episode.get("title"),
            "duration_s": _num(episode.get("duration_s")),
            "trim_start_s": trim_start_s,
            "num_segments": len(final_json.get("segments", [])) if isinstance(final_json.get("segments"), list) else None,
            "num_turns": len(final_json.get("speaker_turns", [])) if isinstance(final_json.get("speaker_turns"), list) else None,
            "num_overlap_candidates": len(final_json.get("overlap_candidates", [])) if isinstance(final_json.get("overlap_candidates"), list) else None,
            "participants_record_count": len(participants) if isinstance(participants, list) else None,
        },
        "participants_provenance": {
            "record_count": len(participants) if isinstance(participants, list) else None,
            "note": ("final.json participants is empty because no roster/roles artifacts "
                     "were available when finalize ran; host identification does not depend on them."),
        },
        **result,
        "limitations": [
            "Single-episode heuristic; not validated for generalization.",
            "Diarization labels are not stable identities and are never renamed.",
            "ASR errors propagate into question/address detection.",
            "Scores are heuristic and uncalibrated; confidence_band is coarse, not probabilistic.",
            "Phrases are discovered only from this episode's transcript.",
        ],
    }
