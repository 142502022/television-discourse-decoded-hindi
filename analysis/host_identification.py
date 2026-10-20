#!/usr/bin/env python3
"""
Standalone heuristic host/moderator identification for the MTP pipeline.

Reads:
    data/v2/videos/<video_id>/final.json

Writes:
    data/v2/videos/<video_id>/host_identification/host_identification.json

Scores are transparent heuristic ranking scores, NOT probabilities.
Manually verify predictions, especially before using them in research.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_VIDEOS_DIR = Path("data/v2/videos")

QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "will", "should", "have",
    "has", "tell", "explain", "answer",
}

# Generic moderation cues. Keep these separate from opening/closing cues.
MODERATOR_PHRASES = (
    "let me ask", "my question", "i want to ask", "let me bring in",
    "let me go to", "joining us", "welcome to the show", "good evening",
    "one at a time", "please respond", "quick response", "final word",
    "let me widen this", "let me take that", "let's be very clear",
    "let us be very clear", "over to you", "your response",
)

OPENING_PHRASES = (
    "let me widen this", "the question of the day", "welcome to",
    "welcome back", "good evening", "good morning", "today we are discussing",
    "tonight we are discussing", "our face-off", "joining me",
    "joining us", "with me is", "let me introduce",
)

GUEST_INTRO_PHRASES = (
    "joining me", "joining us", "with me is", "with us is",
    "is joining me", "is joining us", "national spokesperson",
    "our guest", "let me introduce", "we have with us",
)

CLOSING_PHRASES = (
    "thanks very much for watching", "thank you for watching",
    "thanks for watching", "thanks very much for joining us",
    "thank you for joining us", "stay safe", "stay well",
    "have a safe", "good night", "goodbye", "namaskar", "jai hind",
    "that's all the time we have", "that is all the time we have",
    "we leave it there", "we have to leave it there",
)

FLOOR_MANAGEMENT_PHRASES = (
    "let me bring in", "let me go to", "over to you", "your response",
    "please respond", "quick response", "final word", "one at a time",
    "i want to hear from", "let's hear from", "let us hear from",
    "now to", "turn to",
)

ADDRESS_TOKENS = {
    "mr", "mrs", "ms", "sir", "madam", "minister", "doctor", "dr",
    "spokesperson", "panelist",
}


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_valid_turns(data):
    turns = []
    for item in data.get("speaker_turns", []):
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        start = safe_float(item.get("start"), -1)
        end = safe_float(item.get("end"), -1)
        if speaker and start >= 0 and end > start:
            turns.append({"speaker": str(speaker), "start": start, "end": end})
    return sorted(turns, key=lambda item: (item["start"], item["end"]))


def load_valid_segments(data):
    segments = []
    for item in data.get("segments", []):
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        start = safe_float(item.get("start"), -1)
        end = safe_float(item.get("end"), -1)
        text = clean_text(item.get("text"))
        if speaker and start >= 0 and end > start and text:
            segments.append({
                "speaker": str(speaker),
                "start": start,
                "end": end,
                "text": text,
            })
    return sorted(segments, key=lambda item: (item["start"], item["end"]))


def is_question(text):
    text = clean_text(text).lower()
    if "?" in text:
        return True
    words = re.findall(r"[a-z']+", text)
    return bool(words and words[0] in QUESTION_STARTERS)


def contains_any(text, phrases):
    lowered = clean_text(text).lower()
    return any(phrase in lowered for phrase in phrases)


def contains_moderator_phrase(text):
    return contains_any(text, MODERATOR_PHRASES)


def has_addressing_cue(text):
    lowered = clean_text(text).lower()
    words = set(re.findall(r"[a-z']+", lowered))
    return bool(words & ADDRESS_TOKENS) or bool(
        re.search(r"\b(you|your|yourself)\b", lowered)
    )


def calculate_features(turns, segments, opening_window_seconds=90.0,
                       closing_window_seconds=90.0):
    stats = defaultdict(lambda: {
        "speaking_seconds": 0.0,
        "turn_count": 0,
        "question_count": 0,
        "addressing_cue_count": 0,
        "moderator_phrase_count": 0,
        "segment_count": 0,
        "opening_framing_count": 0,
        "guest_introduction_count": 0,
        "closing_cue_count": 0,
        "floor_management_count": 0,
        "directed_question_count": 0,
        "between_other_speakers_count": 0,
    })

    # Infer duration from the latest valid segment/turn end.
    latest_end = max(
        [s["end"] for s in segments] + [t["end"] for t in turns],
        default=0.0,
    )
    closing_start = max(0.0, latest_end - closing_window_seconds)

    for turn in turns:
        row = stats[turn["speaker"]]
        row["speaking_seconds"] += turn["end"] - turn["start"]
        row["turn_count"] += 1

    for segment in segments:
        row = stats[segment["speaker"]]
        text = segment["text"]
        row["segment_count"] += 1
        question = is_question(text)
        row["question_count"] += int(question)
        row["addressing_cue_count"] += int(has_addressing_cue(text))
        row["moderator_phrase_count"] += int(contains_moderator_phrase(text))

        if segment["start"] <= opening_window_seconds:
            row["opening_framing_count"] += int(
                contains_any(text, OPENING_PHRASES)
            )
            row["guest_introduction_count"] += int(
                contains_any(text, GUEST_INTRO_PHRASES)
            )

        if segment["start"] >= closing_start:
            row["closing_cue_count"] += int(
                contains_any(text, CLOSING_PHRASES)
            )

        floor_cue = contains_any(text, FLOOR_MANAGEMENT_PHRASES)
        row["floor_management_count"] += int(floor_cue)
        # This is only a weak textual proxy: a question containing a name or
        # floor-management phrase is not guaranteed to be directed to a guest.
        row["directed_question_count"] += int(question and floor_cue)

    # Rough interaction signal. Overlapping turns can make this imperfect.
    ordered = sorted(turns, key=lambda item: (item["start"], item["end"]))
    for index in range(1, len(ordered) - 1):
        current = ordered[index]
        previous = ordered[index - 1]
        following = ordered[index + 1]
        if (
            current["speaker"] != previous["speaker"]
            and current["speaker"] != following["speaker"]
            and previous["speaker"] != following["speaker"]
        ):
            stats[current["speaker"]]["between_other_speakers_count"] += 1

    # Convert each measure to a 0..1 relative score within this episode.
    # Opening, guest-introduction and closing features are separate so they
    # can be inspected and validated before final weights are tuned.
    signal_names = (
        "speaking_seconds",
        "turn_count",
        "question_count",
        "addressing_cue_count",
        "moderator_phrase_count",
        "between_other_speakers_count",
        "opening_framing_count",
        "guest_introduction_count",
        "closing_cue_count",
        "floor_management_count",
        "directed_question_count",
    )
    maxima = {
        name: max((row.get(name, 0) for row in stats.values()), default=0)
        for name in signal_names
    }

    # Initial experimental weights, not validated. Role-specific evidence is
    # prioritized over generic participation. Tune only after checking several
    # episodes with known host identities.
    weights = {
        "speaking_seconds": 0.05,
        "turn_count": 0.05,
        "question_count": 0.08,
        "addressing_cue_count": 0.05,
        "moderator_phrase_count": 0.10,
        "between_other_speakers_count": 0.07,
        "opening_framing_count": 0.15,
        "guest_introduction_count": 0.15,
        "closing_cue_count": 0.15,
        "floor_management_count": 0.10,
        "directed_question_count": 0.05,
    }

    candidates = []
    total_speech = sum(row["speaking_seconds"] for row in stats.values()) or 1.0

    for speaker, row in stats.items():
        signals = {}
        score = 0.0
        for name in signal_names:
            maximum = maxima[name]
            normalized = (row.get(name, 0) / maximum) if maximum else 0.0
            signals[name] = round(normalized, 4)
            score += weights[name] * normalized

        candidates.append({
            "speaker": speaker,
            "ranking_score": round(score, 4),
            "speech_share_percent": round(
                100 * row["speaking_seconds"] / total_speech, 2
            ),
            "features": {
                name: (
                    round(row.get(name, 0), 2)
                    if name == "speaking_seconds"
                    else row.get(name, 0)
                )
                for name in signal_names
            },
            "normalized_signals": signals,
            "weights_used": weights,
        })

    candidates.sort(key=lambda item: item["ranking_score"], reverse=True)
    return candidates


def analyze_episode(json_path, top_n=3, opening_window_seconds=90.0,
                    closing_window_seconds=90.0):
    data = read_json(json_path)
    turns = load_valid_turns(data)
    segments = load_valid_segments(data)
    candidates = calculate_features(
        turns, segments,
        opening_window_seconds=opening_window_seconds,
        closing_window_seconds=closing_window_seconds,
    )

    video_id = data.get("episode", {}).get("video_id") or json_path.parent.name
    title = data.get("episode", {}).get("title", "")

    if not candidates:
        prediction = None
        confidence_band = "uncertain"
        margin = None
    else:
        prediction = candidates[0]["speaker"]
        margin = round(
            candidates[0]["ranking_score"] -
            (candidates[1]["ranking_score"] if len(candidates) > 1 else 0.0),
            4,
        )
        if len(candidates) < 2:
            confidence_band = "uncertain"
        elif margin >= 0.15:
            confidence_band = "higher_heuristic_separation"
        elif margin >= 0.06:
            confidence_band = "moderate_heuristic_separation"
        else:
            confidence_band = "low_heuristic_separation"

    return {
        "video_id": video_id,
        "title": title,
        "source_file": str(json_path),
        "method": "experimental transparent heuristic; no trained classifier",
        "predicted_host_speaker": prediction,
        "predicted_host_candidate_rank": 1 if prediction is not None else None,
        "confidence_band": confidence_band,
        "top_score_margin": margin,
        "analysis_windows_seconds": {
            "opening": opening_window_seconds,
            "closing": closing_window_seconds,
        },
        "warning": (
            "Scores and confidence bands are heuristic, not calibrated "
            "probabilities. Opening/closing phrase lists are incomplete and "
            "currently English-oriented; translated ASR text can affect cues. "
            "Manually verify predictions and evaluate across known-host episodes."
        ),
        "candidate_count": len(candidates),
        "candidate_policy": (
            "Candidates are ranked by heuristic score. Rank 1 is the current "
            "predicted host; ranks 2 and 3 are fallback candidates for manual review."
        ),
        "candidates": [
            {
                **candidate,
                "rank": rank,
                "is_predicted_host": rank == 1,
            }
            for rank, candidate in enumerate(candidates[:top_n], start=1)
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify likely host/moderator speakers from final.json files."
    )
    parser.add_argument(
        "--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR,
        help="Directory containing <video_id>/final.json (default: data/v2/videos)",
    )
    parser.add_argument(
        "--video-id", help="Analyze only this video ID; omit to analyze all episodes."
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Number of candidates to include in each output (default: 3).",
    )
    parser.add_argument(
        "--opening-window", type=float, default=90.0,
        help="Seconds from episode start used for opening cues (default: 90).",
    )
    parser.add_argument(
        "--closing-window", type=float, default=90.0,
        help="Seconds from episode end used for closing cues (default: 90).",
    )
    args = parser.parse_args()

    if not args.videos_dir.exists():
        raise SystemExit(
            f"Videos directory not found: {args.videos_dir.resolve()}\n"
            "Run this script from your repository root, or pass --videos-dir."
        )

    if args.video_id:
        json_files = [args.videos_dir / args.video_id / "final.json"]
        if not json_files[0].is_file():
            raise SystemExit(f"Could not find episode file: {json_files[0]}")
    else:
        json_files = sorted(args.videos_dir.glob("*/final.json"))

    if not json_files:
        raise SystemExit(f"No final.json files found under {args.videos_dir}")

    for json_path in json_files:
        result = analyze_episode(
            json_path,
            top_n=max(1, args.top_n),
            opening_window_seconds=max(0.0, args.opening_window),
            closing_window_seconds=max(0.0, args.closing_window),
        )
        output_dir = json_path.parent / "host_identification"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "host_identification.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 65)
        print(f"Episode: {result['video_id']}")
        print(f"Title: {result['title']}")
        print(f"Predicted host: {result['predicted_host_speaker']}")
        print(f"Heuristic separation: {result['confidence_band']}")
        print(f"Saved: {output_path}")
        print("\nTop candidates:")
        for rank, candidate in enumerate(result["candidates"], start=1):
            features = candidate["features"]
            print(
                f"{rank}. {candidate['speaker']}: "
                f"score={candidate['ranking_score']:.3f}, "
                f"speech={candidate['speech_share_percent']:.1f}%, "
                f"turns={features['turn_count']}, "
                f"questions={features['question_count']}, "
                f"opening={features['opening_framing_count']}, "
                f"guest_intro={features['guest_introduction_count']}, "
                f"closing={features['closing_cue_count']}, "
                f"floor={features['floor_management_count']}"
            )


if __name__ == "__main__":
    main()