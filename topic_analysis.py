"""Sample-first semantic topic analysis for v2 final JSON artifacts.

Topic boundaries come from transcript embeddings. Overlaps and interruptions are
reported as contextual observations only, never used to create a boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VERSION = "0.1.0"
STOP = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
        "their", "this", "to", "was", "were", "with", "you", "your", "we",
        "they", "what", "why", "how", "not", "do", "does", "can", "about"}


@dataclass(frozen=True)
class TopicConfig:
    model_name: str = MODEL_NAME
    language: str = "en"
    context_radius: int = 1
    subtopic_boundary_quantile: float = 0.15
    broad_boundary_quantile: float = 0.05
    checkpoint_seconds: float = 300.0
    restoration_lookahead_segments: int = 2


def finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def r3(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def issue(code: str, message: str, count: int = 1) -> dict[str, Any]:
    return {"code": code, "message": message, "count": count}


def identifier(prefix: str, *parts: str) -> str:
    return prefix + "_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def load_episodes(root: Path, language: str) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    loaded, skipped, failed = [], [], []
    if not root.is_dir():
        return loaded, skipped, [{"path": str(root), "reason": "input_directory_missing"}]
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "final.json"
        if not path.is_file():
            skipped.append({"video_id": directory.name, "reason": "missing_final_json"})
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            failed.append({"video_id": directory.name, "reason": "invalid_json", "detail": str(exc)})
            continue
        asr = data.get("pipeline", {}).get("asr", {}) if isinstance(data, dict) else {}
        if not isinstance(data, dict):
            failed.append({"video_id": directory.name, "reason": "invalid_top_level_schema"})
        elif language and asr.get("language") != language:
            skipped.append({"video_id": directory.name, "reason": "language_filtered", "language": asr.get("language")})
        else:
            loaded.append((directory.name, data))
    return loaded, skipped, failed


def unit_speaker(segment: dict[str, Any]) -> str:
    if isinstance(segment.get("speaker"), str):
        return segment["speaker"]
    words = segment.get("words") if isinstance(segment.get("words"), list) else []
    labels = Counter(word.get("speaker") for word in words if isinstance(word, dict) and isinstance(word.get("speaker"), str))
    return labels.most_common(1)[0][0] if labels else "UNKNOWN"


def extract_units(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw, units, warnings, bad = data.get("segments"), [], [], 0
    if not isinstance(raw, list):
        return [], [issue("missing_or_invalid_segments", "segments is not a list")]
    for index, segment in enumerate(raw):
        if not isinstance(segment, dict):
            bad += 1
            continue
        text, start, end = segment.get("text"), finite(segment.get("start")), finite(segment.get("end"))
        if not isinstance(text, str) or not text.strip():
            warnings.append(issue("missing_transcript_text", "A segment lacks transcript text"))
        elif start is None or end is None or end <= start:
            bad += 1
        else:
            units.append({"unit_id": f"unit_{index:05d}", "source_segment_index": index,
                          "start": start, "end": end, "speaker": unit_speaker(segment), "text": text.strip()})
    if bad:
        warnings.append(issue("invalid_segment_time_excluded", "Segment has invalid timestamp", bad))
    if any(units[i]["start"] < units[i - 1]["start"] for i in range(1, len(units))):
        warnings.append(issue("segments_out_of_order", "Segments sorted chronologically for analysis"))
    return sorted(units, key=lambda unit: (unit["start"], unit["end"], unit["unit_id"])), warnings


def context_texts(units: list[dict[str, Any]], radius: int) -> list[str]:
    return [" ".join(unit["text"] for unit in units[max(0, i - radius):min(len(units), i + radius + 1)])
            for i in range(len(units))]


def embed_texts(texts: list[str], model_name: str) -> tuple[list[list[float]], dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers before running topic analysis.") from exc
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = [list(map(float, vector)) for vector in vectors]
    return vectors, {"model": model_name, "embedding_dimensions": len(vectors[0]) if vectors else 0, "normalization": "L2"}


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def quantile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def boundaries(embeddings: list[list[float]], fraction: float) -> tuple[list[int], list[dict[str, Any]]]:
    sims = [cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    threshold = quantile(sims, fraction)
    found = []
    for index, score in enumerate(sims):
        before = sims[index - 1] if index else float("inf")
        after = sims[index + 1] if index + 1 < len(sims) else float("inf")
        if threshold is not None and score <= threshold and score <= before and score <= after:
            found.append(index + 1)
    return found, [{"boundary_unit_index": i + 1, "adjacent_context_similarity": r3(score), "semantic_change_score": r3(1 - score)}
                   for i, score in enumerate(sims)]


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    result = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm else result


def make_segments(units: list[dict[str, Any]], embeddings: list[list[float]], points: list[int], level: str, video_id: str) -> list[dict[str, Any]]:
    starts, output = [0] + sorted(set(points)) + [len(units)], []
    for left, right in zip(starts, starts[1:]):
        selected = units[left:right]
        if not selected:
            continue
        text = " ".join(unit["text"] for unit in selected)
        output.append({"segment_id": identifier(level, video_id, str(left), str(right), text[:120]),
                       "unit_start_index": left, "unit_end_index_exclusive": right,
                       "start": selected[0]["start"], "end": selected[-1]["end"],
                       "speakers": sorted({unit["speaker"] for unit in selected}),
                       "unit_ids": [unit["unit_id"] for unit in selected], "excerpt": text[:600],
                       "_embedding": centroid(embeddings[left:right]), "review_status": "candidate",
                       "boundary_provenance": "sentence_embedding_local_similarity"})
    return output


def label_segments(records: list[dict[str, Any]]) -> None:
    """Candidate labels use corpus evidence, never a predefined debate taxonomy."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        matrix = vectorizer.fit_transform([record["excerpt"] for record in records])
        terms = vectorizer.get_feature_names_out()
        for i, record in enumerate(records):
            row = matrix.getrow(i)
            top = sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], terms[pair[0]]))[:3]
            record["candidate_label"] = " · ".join(terms[column] for column, _ in top) or "Unlabeled transcript segment"
            record["label_evidence_terms"] = [terms[column] for column, _ in top]
    except ValueError:
        for record in records:
            tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", record["excerpt"]) if token.lower() not in STOP]
            record["candidate_label"] = " · ".join(tokens[:3]) if tokens else "Unlabeled transcript segment"
            record["label_evidence_terms"] = tokens[:3]


def assign_hierarchy(broad: list[dict[str, Any]], subs: list[dict[str, Any]], video_id: str) -> None:
    label_segments(broad)
    label_segments(subs)
    for i, record in enumerate(broad, 1):
        record["broad_topic_id"] = f"broad_{video_id}_{i:03d}"
        record["candidate_definition"] = record["excerpt"][:300]
        record.pop("_embedding", None)
    for i, record in enumerate(subs, 1):
        record["subtopic_id"] = f"subtopic_{video_id}_{i:03d}"
        parent = next((item for item in broad if item["start"] <= record["start"] and item["end"] >= record["end"]), None)
        if parent is None:
            parent = min(broad, key=lambda item: abs(item["start"] - record["start"]))
        record["broad_topic_id"], record["broad_topic_label"] = parent["broad_topic_id"], parent["candidate_label"]
        record["candidate_definition"] = record["excerpt"][:300]
        record.pop("_embedding", None)


def valid_interruptions(data: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for i, event in enumerate(data.get("interruptions", []) if isinstance(data.get("interruptions"), list) else []):
        if not isinstance(event, dict):
            continue
        start, end = finite(event.get("start")), finite(event.get("end"))
        if start is not None and end is not None and end > start:
            output.append({"interruption_id": f"interruption_{i:04d}", "start": start, "end": end,
                           "interrupter": event.get("interrupter"), "speakers": event.get("speakers", [])})
    return output


def intersects(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def shift_layers(subs: list[dict[str, Any]], interruptions: list[dict[str, Any]], lookahead: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events, used = [], set()
    for i, (before, after) in enumerate(zip(subs, subs[1:]), 1):
        related = [item for item in interruptions if intersects(item["start"], item["end"], after["start"], after["end"])]
        used.update(item["interruption_id"] for item in related)
        later = subs[i + 1:i + 1 + lookahead]
        restored = any(item["subtopic_id"] == before["subtopic_id"] for item in later)
        direction = "same_broad_topic" if before["broad_topic_id"] == after["broad_topic_id"] else "different_broad_topic"
        label = "topic_restoration" if restored else ("potential_topic_drift" if direction == "different_broad_topic" else "topic_transition")
        events.append({"event_id": f"shift_{i:04d}", "timestamp": after["start"],
                       "attempting_speaker": after["speakers"][0] if len(after["speakers"]) == 1 else None,
                       "original": {"broad_topic_id": before["broad_topic_id"], "subtopic_id": before["subtopic_id"], "label": before["candidate_label"]},
                       "candidate_new": {"broad_topic_id": after["broad_topic_id"], "subtopic_id": after["subtopic_id"], "label": after["candidate_label"]},
                       "attempt_status": "possible_semantic_shift", "immediate_direction": after["subtopic_id"],
                       "subsequent_direction": later[0]["subtopic_id"] if later else None,
                       "original_subtopic_restored": restored, "relationship_label": label,
                       "interruption_observed": bool(related), "related_interruptions": related,
                       "resistance_status": "indeterminate", "provenance": "model_inferred",
                       "confidence_status": "review_required",
                       "evidence": {"before_excerpt": before["excerpt"][:240], "after_excerpt": after["excerpt"][:240]}})
    contextual = [{**item,
                   "linked_topic_shift_event": next((event["event_id"] for event in events if item["interruption_id"] in {x["interruption_id"] for x in event["related_interruptions"]}), None),
                   "topic_relationship": "related_to_possible_shift" if item["interruption_id"] in used else "no_semantic_boundary_intersection",
                   "provenance": "observed_temporal_intersection"} for item in interruptions]
    return events, contextual


def checkpoints(units: list[dict[str, Any]], subs: list[dict[str, Any]], shifts: list[dict[str, Any]], duration_s: Optional[float], seconds: float) -> list[dict[str, Any]]:
    output, start = [], 0.0
    while duration_s and start < duration_s:
        end = min(duration_s, start + seconds)
        active = [item for item in subs if intersects(item["start"], item["end"], start, end)]
        weight = lambda item: max(0, min(end, item["end"]) - max(start, item["start"]))
        dominant = max(active, key=lambda item: (weight(item), item["subtopic_id"])) if active else None
        output.append({"window_start": start, "window_end": end,
                       "dominant_broad_topic": {"id": dominant["broad_topic_id"], "label": dominant["broad_topic_label"]} if dominant else None,
                       "subtopics": [{"id": item["subtopic_id"], "label": item["candidate_label"]} for item in active],
                       "speakers": sorted({unit["speaker"] for unit in units if intersects(unit["start"], unit["end"], start, end)}),
                       "topic_changes": [item["event_id"] for item in shifts if start <= item["timestamp"] < end],
                       "continues_previous_subtopic": bool(active and active[0]["start"] < start),
                       "summary_evidence": dominant["excerpt"][:240] if dominant else None,
                       "uncertainty": "review_required" if active else "no_transcript_coverage"})
        start = end
    return output


def analyze_episode(video_id: str, data: dict[str, Any], config: TopicConfig) -> dict[str, Any]:
    units, warnings = extract_units(data)
    episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
    duration = finite(episode.get("duration_s"))
    empty = {"metadata": {"video_id": video_id, "title": episode.get("title"), "duration_s": duration},
             "units": units, "broad_topics": [], "subtopics": [], "five_minute_checkpoints": [],
             "topic_shift_events": [], "contextual_interruptions": [], "boundary_evidence": [], "warnings": warnings}
    if len(units) < 2:
        empty["warnings"].append(issue("insufficient_transcript_context", "At least two timed segments are required"))
        return empty
    vectors, model = embed_texts(context_texts(units, config.context_radius), config.model_name)
    broad_points, evidence = boundaries(vectors, config.broad_boundary_quantile)
    sub_points, _ = boundaries(vectors, config.subtopic_boundary_quantile)
    broad, subs = make_segments(units, vectors, broad_points, "broad", video_id), make_segments(units, vectors, sorted(set(sub_points) | set(broad_points)), "subtopic", video_id)
    assign_hierarchy(broad, subs, video_id)
    shifts, contextual = shift_layers(subs, valid_interruptions(data), config.restoration_lookahead_segments)
    return {**empty, "metadata": {**empty["metadata"], "model": model}, "broad_topics": broad, "subtopics": subs,
            "five_minute_checkpoints": checkpoints(units, subs, shifts, duration, config.checkpoint_seconds),
            "topic_shift_events": shifts, "contextual_interruptions": contextual, "boundary_evidence": evidence,
            "warnings": warnings + [issue("candidate_labels_require_review", "All topic labels and shift interpretations require human review")]}


def choose_sample(records: list[tuple[str, dict[str, Any]]], count: int) -> list[tuple[str, dict[str, Any]]]:
    if len(records) <= count:
        return records
    def duration(item: tuple[str, dict[str, Any]]) -> float:
        return finite(item[1].get("episode", {}).get("duration_s")) or 0.0
    ordered, positions = sorted(records, key=lambda item: (duration(item), item[0])), sorted({0, len(records) // 2, len(records) - 1})
    return [ordered[position] for position in positions][:count]


def vocabulary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for level, field, key in (("broad", "broad_topics", "broad_topic_id"), ("subtopic", "subtopics", "subtopic_id")):
        records = [record for episode in episodes for record in episode[field]]
        for record in records:
            duplicates = [other[key] for other in records if other[key] != record[key] and other["candidate_label"].lower() == record["candidate_label"].lower()]
            entries.append({"topic_id": record[key], "level": level, "candidate_name": record["candidate_label"],
                            "canonical_name": None, "definition": record["candidate_definition"],
                            "parent_topic_id": record.get("broad_topic_id") if level == "subtopic" else None,
                            "representative_example": record["excerpt"][:240],
                            "duplicate_label_mappings": duplicates, "review_status": "candidate"})
    return {"entries": entries, "consolidation_policy": "Exact duplicate labels are review suggestions; no semantic merge is automatic."}


def build_report(videos_dir: Path, config: TopicConfig, sample_size: int = 3, all_episodes: bool = False) -> dict[str, Any]:
    loaded, skipped, failed = load_episodes(videos_dir, config.language)
    selected = loaded if all_episodes else choose_sample(loaded, sample_size)
    selected_ids = {item[0] for item in selected}
    if not all_episodes:
        skipped += [{"video_id": video_id, "reason": "not_in_initial_sample"} for video_id, _ in loaded if video_id not in selected_ids]
    episodes = []
    for video_id, data in selected:
        try:
            episodes.append(analyze_episode(video_id, data, config))
        except (RuntimeError, TypeError, ValueError) as exc:
            failed.append({"video_id": video_id, "reason": "analysis_failed", "detail": str(exc)})
    warning_counts = Counter(item["code"] for episode in episodes for item in episode["warnings"])
    return {"report_metadata": {"analysis_version": VERSION, "analysis_scope": "full_dataset" if all_episodes else "initial_representative_sample",
                                "config": asdict(config), "method": "Contextual sentence embeddings with local semantic-similarity valleys.",
                                "semantic_policy": "Labels are candidate vocabulary entries; model outputs are not observed facts."},
            "episodes": episodes, "topic_vocabulary": vocabulary(episodes),
            "aggregate": {"processing": {"analyzed": len(episodes), "skipped": len(skipped), "failed": len(failed)},
                          "topic_segments": {"broad_total": sum(len(item["broad_topics"]) for item in episodes), "subtopic_total": sum(len(item["subtopics"]) for item in episodes)},
                          "shift_events": {"total": sum(len(item["topic_shift_events"]) for item in episodes),
                                           "with_observed_interruption": sum(sum(event["interruption_observed"] for event in item["topic_shift_events"]) for item in episodes)},
                          "quality_warning_counts": dict(sorted(warning_counts.items()))},
            "skipped": skipped, "failed": failed,
            "review_template": {"boundary_overrides": [{"video_id": "", "timestamp_s": None, "action": "add_or_remove", "rationale": ""}],
                                "label_overrides": [{"topic_id": "", "approved_name": "", "approved_definition": "", "reviewer": ""}],
                                "shift_event_overrides": [{"event_id": "", "relationship_label": "indeterminate", "restoration": None, "rationale": "", "reviewer": ""}]},
            "limitations": ["English model baseline; non-English artifacts are filtered by default.", "ASR/diarization errors affect evidence.", "Five-minute windows never force boundaries.", "Interruption links are temporal context, not causal evidence.", "No accuracy claim without human annotations."]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-first semantic topic analysis from v2 debate artifacts.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/topic_analysis/topic_analysis_sample.json"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--subtopic-boundary-quantile", type=float, default=0.15)
    parser.add_argument("--broad-boundary-quantile", type=float, default=0.05)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.sample_size < 1 or args.context_radius < 0 or args.checkpoint_seconds <= 0:
        parser.error("sample-size must be positive, context-radius non-negative, checkpoint-seconds positive.")
    if not 0 <= args.subtopic_boundary_quantile <= 1 or not 0 <= args.broad_boundary_quantile <= 1:
        parser.error("boundary quantiles must be between 0 and 1.")
    config = TopicConfig(args.model, args.language, args.context_radius, args.subtopic_boundary_quantile,
                         args.broad_boundary_quantile, args.checkpoint_seconds, 2)
    report = build_report(args.videos_dir, config, args.sample_size, args.all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    status = report["aggregate"]["processing"]
    print(f"Topic analysis: {status['analyzed']} analyzed, {status['skipped']} skipped, {status['failed']} failed")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()

