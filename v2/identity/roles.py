"""Milestone 5, part 2: heuristic role inference (anchor vs participant).

Uses discourse signals already present in our artifacts to label each
diarized speaker by their function in the debate: the anchor opens the show,
holds the largest share of speech, asks most questions, addresses others
directly, and initiates more interruptions than they receive.

This is a transparent, feature-based baseline: every decision carries an
``evidence`` block with the raw features. Named identity (matching these labels
to people) is layered *on top of* this via speaker embeddings.
"""

from typing import Any, Dict, List, Optional

QUESTION_STARTERS = {
    "what",
    "why",
    "how",
    "who",
    "whom",
    "whose",
    "which",
    "when",
    "where",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "will",
    "would",
    "should",
    "shall",
    "has",
    "have",
    "had",
}


def segment_is_question(segment: Dict[str, Any]) -> bool:
    text = (segment.get("text") or "").strip()
    if text.endswith("?"):
        return True
    first_word = (text.split() or [""])[0].lower()
    return first_word in QUESTION_STARTERS


def segment_addresses_other(segment: Dict[str, Any]) -> bool:
    words = {(w.get("word") or "").lower().strip(".,!?") for w in segment.get("words") or []}
    if not words:
        words = {s.lower().strip(".,!?\"' ") for s in (segment.get("text") or "").split()}
    return bool({"you", "your", "yours", "u", "sir", "madam"} & words)


def speaker_features(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-speaker discourse features derived from diarized segments."""
    features: Dict[str, Dict[str, float]] = {}

    def ensure(speaker: str) -> Dict[str, Any]:
        if speaker not in features:
            features[speaker] = {
                "first_start": float("inf"),
                "last_end": 0.0,
                "speech_seconds": 0.0,
                "segment_count": 0,
                "questions": 0,
                "addresses": 0,
            }
        return features[speaker]

    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        feat = ensure(speaker)
        start, end = segment["start"], segment["end"]
        feat["first_start"] = min(feat["first_start"], start)
        feat["last_end"] = max(feat["last_end"], end)
        feat["speech_seconds"] += end - start
        feat["segment_count"] += 1
        if segment_is_question(segment):
            feat["questions"] += 1
        if segment_addresses_other(segment):
            feat["addresses"] += 1
    return features


def interruption_totals(
    interruption_stats: Optional[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """Per-speaker initiated/received interruption counts from OSD analysis."""
    totals: Dict[str, Dict[str, int]] = {}
    for row in interruption_stats or []:
        initiator = totals.setdefault(row["interrupter"], {"initiated": 0, "received": 0})
        initiator["initiated"] += row["count"]
        receiver = totals.setdefault(row["victim"], {"initiated": 0, "received": 0})
        receiver["received"] += row["count"]
    return totals


def infer_roles(
    segments: List[Dict[str, Any]],
    interruption_stats: Optional[List[Dict[str, Any]]] = None,
    namer: Any = lambda speaker: speaker,
) -> Dict[str, Any]:
    """Label speakers as ``anchor`` or ``participant`` with evidence.

    ``namer`` optionally maps a speaker label to a display name (identity
    linking); it defaults to the raw label.
    """
    features = speaker_features(segments)
    if not features:
        return {"method": "discourse-heuristic", "roles": {}, "anchor": None}

    total_speech = sum(f["speech_seconds"] for f in features.values())
    first_speaker = min(features, key=lambda s: features[s]["first_start"])
    interruptions = interruption_totals(interruption_stats)

    scores: Dict[str, float] = {}
    for speaker, feat in features.items():
        share = feat["speech_seconds"] / total_speech if total_speech else 0.0
        inter = interruptions.get(speaker, {"initiated": 0, "received": 0})
        score = (
            1.0  # base
            + (1.0 if speaker == first_speaker else 0.0)
            + share
            + min(feat["questions"], 4) * 0.5
            + (0.5 if feat["addresses"] else 0.0)
            + max(-0.6, min(0.6, (inter["initiated"] - inter["received"]) * 0.2))
        )
        scores[speaker] = score

    total_score = sum(scores.values()) or 1.0
    anchor = max(scores, key=lambda s: scores[s])
    roles: Dict[str, Any] = {}
    for speaker in sorted(scores):
        roles[speaker] = {
            "role": "anchor" if speaker == anchor else "participant",
            "score": round(scores[speaker], 3),
            "confidence": round(scores[speaker] / total_score, 3),
            "name": namer(speaker),
            "evidence": {
                "first_start_s": feat_round(features[speaker]["first_start"]),
                "speech_seconds": feat_round(features[speaker]["speech_seconds"]),
                "speech_share": feat_round(features[speaker]["speech_seconds"] / total_speech) if total_speech else 0.0,
                "segment_count": features[speaker]["segment_count"],
                "questions": features[speaker]["questions"],
                "addresses": features[speaker]["addresses"],
                "interruptions_initiated": interruptions.get(speaker, {}).get("initiated", 0),
                "interruptions_received": interruptions.get(speaker, {}).get("received", 0),
            },
        }

    return {
        "method": "discourse-heuristic",
        "total_speech_s": feat_round(total_speech),
        "anchor": namer(anchor),
        "anchor_speaker": anchor,
        "roles": roles,
    }


def feat_round(value: float) -> float:
    return round(float(value), 3)