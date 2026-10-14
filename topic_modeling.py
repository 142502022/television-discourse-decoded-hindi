"""Corpus-wide hierarchical topic and subtopic modeling for v2 debate artifacts.

A single BERTopic model is fitted across every transcript unit of the selected
(English) episodes. Subtopics are BERTopic's discovered base topics; broad
topics are obtained by agglomeratively merging those base topics at a coarser
resolution. BERTopic's own ``hierarchical_topics`` tree is exported as evidence.
Every unit is then assigned back to its episode as a topic timeline.

Model outputs are labeled candidate topics, never observed ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VERSION = "0.1.0"


@dataclass(frozen=True)
class TopicConfig:
    model_name: str = MODEL_NAME
    language: str = "en"
    window_seconds: float = 45.0
    min_window_seconds: float = 12.0
    max_window_chars: int = 1500
    min_topic_size: int = 8
    n_broad_topics: int = 8
    hdbscan_selection: str = "leaf"
    umap_neighbors: int = 15
    umap_components: int = 5
    reduce_outliers: bool = True
    random_state: int = 42
    trim_config: Optional[str] = None
    drop_unreliable: bool = True
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6


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
    loaded: list[tuple[str, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
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
        if not isinstance(data, dict):
            failed.append({"video_id": directory.name, "reason": "invalid_top_level_schema"})
            continue
        asr = data.get("pipeline", {}).get("asr", {})
        if language and asr.get("language") != language:
            skipped.append({"video_id": directory.name, "reason": "language_filtered", "language": asr.get("language")})
        else:
            loaded.append((directory.name, data))
    return loaded, skipped, failed


def load_trim_map(path: Optional[str]) -> dict[str, dict[str, Any]]:
    """Read a per-video ad/intro trim map.

    Accepts ``{"id": 720}`` or ``{"id": {"trim_start_s": 720, "reason": "..."}}``.
    Returns ``{video_id: {"trim_start_s": float, "reason": str|None}}``.
    """
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for video_id, value in raw.items():
        if isinstance(value, dict):
            seconds = finite(value.get("trim_start_s"))
            reason = value.get("reason") if isinstance(value.get("reason"), str) else None
        else:
            seconds = finite(value)
            reason = None
        if seconds is not None and seconds > 0:
            result[str(video_id)] = {"trim_start_s": seconds, "reason": reason}
    return result


def build_units(
    segments: list[Any],
    window_s: float,
    min_window_s: float,
    max_chars: int,
    *,
    trim_start_s: float = 0.0,
    min_avg_logprob: Optional[float] = None,
    max_no_speech_prob: Optional[float] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Merge consecutive timed segments into windows of roughly window_s seconds.

    Single debate turns are too short for stable topical signal; windows give the
    embedding model enough lexical context. Segments inside a configured ad/intro
    region (``trim_start_s``) and low-confidence segments (Whisper logprob /
    no-speech) are dropped and counted.
    """
    windows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stats = Counter()
    dropped: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    def record_drop(start: Optional[float], end: Optional[float], reason: str) -> None:
        dropped.append({"start": r3(start), "end": r3(end), "reason": reason})

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(item["text"] for item in buffer).strip()
        if not text:
            return
        windows.append({
            "start": buffer[0]["start"],
            "end": buffer[-1]["end"],
            "speakers": sorted({s for item in buffer for s in item["speakers"]}),
            "text": text,
            "source_segment_indices": [item["index"] for item in buffer],
        })

    for index, segment in enumerate(segments if isinstance(segments, list) else []):
        stats["segments_total"] += 1
        if not isinstance(segment, dict):
            stats["dropped_bad_time"] += 1
            continue
        text = segment.get("text")
        start, end = finite(segment.get("start")), finite(segment.get("end"))
        if not isinstance(text, str) or not text.strip():
            stats["dropped_missing_text"] += 1
            warnings.append(issue("missing_transcript_text", "A segment lacks transcript text"))
            record_drop(start, end, "missing_text")
            continue
        if start is None or end is None or end <= start:
            stats["dropped_bad_time"] += 1
            record_drop(start, end, "bad_time")
            continue
        if trim_start_s and end <= trim_start_s:
            stats["dropped_ad_trim"] += 1
            record_drop(start, end, "ad_trim")
            continue
        if min_avg_logprob is not None:
            logprob = finite(segment.get("avg_logprob"))
            if logprob is not None and logprob < min_avg_logprob:
                stats["dropped_low_logprob"] += 1
                record_drop(start, end, "low_logprob")
                continue
        if max_no_speech_prob is not None:
            no_speech = finite(segment.get("no_speech_prob"))
            if no_speech is not None and no_speech > max_no_speech_prob:
                stats["dropped_no_speech"] += 1
                record_drop(start, end, "no_speech")
                continue
        speakers = [segment["speaker"]] if isinstance(segment.get("speaker"), str) else []
        buffer.append({"index": index, "start": start, "end": end, "text": text.strip(), "speakers": speakers})
        span = buffer[-1]["end"] - buffer[0]["start"]
        chars = sum(len(item["text"]) for item in buffer)
        if span >= window_s or chars >= max_chars:
            flush()
            buffer = []
    if buffer:
        span = buffer[-1]["end"] - buffer[0]["start"]
        if span >= min_window_s or not windows:
            flush()
        else:
            windows.append({
                "start": buffer[0]["start"], "end": buffer[-1]["end"],
                "speakers": sorted({s for item in buffer for s in item["speakers"]}),
                "text": " ".join(item["text"] for item in buffer).strip(),
                "source_segment_indices": [item["index"] for item in buffer],
            })
    stats["kept_segments"] = sum(len(window["source_segment_indices"]) for window in windows)
    stats["windows"] = len(windows)
    stats["dropped_ranges"] = dropped
    return windows, warnings, dict(stats)


def embed_texts(texts: list[str], model_name: str) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers before running topic modeling.") from exc
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    return vectors, {"model": model_name, "embedding_dimensions": int(vectors.shape[1]) if vectors.size else 0, "normalization": "L2"}


def group_broad_topics(embeddings: Any, topics: list[int], base_ids: list[int], n_broad: int) -> dict[int, int]:
    """Agglomeratively merge base topics into at most n_broad broad topics."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(base_ids) <= n_broad:
        return {topic: index for index, topic in enumerate(base_ids)}
    centroids = []
    for topic in base_ids:
        members = embeddings[np.array([i for i, t in enumerate(topics) if t == topic])]
        centroid = members.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids.append(centroid / norm if norm else centroid)
    tree = linkage(np.asarray(centroids), method="ward")
    labels = fcluster(tree, t=n_broad, criterion="maxclust")
    return {topic: int(label) for topic, label in zip(base_ids, labels)}


def top_terms(texts: list[str], k: int = 5) -> list[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        scores = matrix.mean(axis=0).A1 if matrix.shape[0] else []
        ranked = sorted(zip(terms, scores), key=lambda pair: (-pair[1], pair[0]))
        return [term for term, _ in ranked[:k]]
    except ValueError:
        counter = Counter(token.lower() for text in texts for token in text.split() if len(token) > 2)
        return [term for term, _ in counter.most_common(k)]


def topic_words(model: Any, topic_id: int, k: int = 5) -> list[str]:
    try:
        return [word for word, _ in model.get_topic(topic_id)[:k]]
    except (TypeError, ValueError):
        return []


def fit_model(docs: list[str], embeddings: Any, config: TopicConfig) -> tuple[Any, list[int], list[float], int]:
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("Install bertopic, umap-learn and hdbscan before running topic modeling.") from exc
    model = BERTopic(
        umap_model=UMAP(n_neighbors=config.umap_neighbors, n_components=config.umap_components,
                        min_dist=0.0, metric="cosine", random_state=config.random_state),
        hdbscan_model=HDBSCAN(min_cluster_size=config.min_topic_size, metric="euclidean",
                              cluster_selection_method=config.hdbscan_selection, prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
        calculate_probabilities=True, verbose=False,
    )
    topics, probabilities = model.fit_transform(docs, embeddings=embeddings)
    initial_outliers = sum(1 for topic in topics if topic == -1)
    if config.reduce_outliers and initial_outliers:
        topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf")
        model.update_topics(docs, topics=topics, vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1))
    if probabilities is not None and getattr(probabilities, "ndim", 1) == 2:
        probabilities = probabilities.max(axis=1)
    probabilities = [float(p) for p in probabilities] if probabilities is not None else [0.0] * len(docs)
    return model, [int(topic) for topic in topics], probabilities, initial_outliers


def build_topic_tables(model: Any, docs: list[str], topics: list[int], embeddings: Any, config: TopicConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int], list[dict[str, Any]]]:
    base_ids = sorted({topic for topic in topics if topic != -1})
    broad_of_topic = group_broad_topics(embeddings, topics, base_ids, config.n_broad_topics)

    subtopics: list[dict[str, Any]] = []
    topic_to_subtopic: dict[int, str] = {}
    for order, topic in enumerate(base_ids, 1):
        members = [docs[i] for i, t in enumerate(topics) if t == topic]
        subtopic_id = f"subtopic_{order:03d}"
        topic_to_subtopic[topic] = subtopic_id
        words = topic_words(model, topic)
        subtopics.append({
            "subtopic_id": subtopic_id, "base_topic_id": topic,
            "candidate_label": " · ".join(words[:3]) or "Unlabeled",
            "evidence_terms": words, "unit_count": len(members),
            "broad_topic_id": f"broad_{broad_of_topic[topic]:03d}",
            "representative_example": members[0][:240] if members else None,
            "review_status": "candidate",
        })

    broad_topics: list[dict[str, Any]] = []
    grouped: dict[int, list[str]] = defaultdict(list)
    for topic in base_ids:
        grouped[broad_of_topic[topic]].extend(docs[i] for i, t in enumerate(topics) if t == topic)
    for group in sorted(grouped):
        terms = top_terms(grouped[group], 5)
        broad_topics.append({
            "broad_topic_id": f"broad_{group:03d}",
            "candidate_label": " · ".join(terms[:3]) or "Unlabeled",
            "evidence_terms": terms, "unit_count": len(grouped[group]),
            "subtopic_ids": [subtopics[i]["subtopic_id"] for i in range(len(subtopics))
                             if subtopics[i]["broad_topic_id"] == f"broad_{group:03d}"],
            "review_status": "candidate",
        })

    try:
        import numpy as np
        hierarchy_frame = model.hierarchical_topics(docs)
        hierarchy = [{
            "parent_id": int(row["Parent_ID"]),
            "child_left_id": int(row["Child_Left_ID"]),
            "child_right_id": int(row["Child_Right_ID"]),
            "distance": r3(float(row["Distance"])),
        } for _, row in hierarchy_frame.iterrows()]
    except Exception:
        hierarchy = []
    return subtopics, broad_topics, topic_to_subtopic, hierarchy


def contiguous_runs(assignments: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for item in assignments:
        if runs and runs[-1]["value"] == item[key]:
            runs[-1]["end"] = item["end"]
            runs[-1]["unit_count"] += 1
            runs[-1]["speakers"] = sorted(set(runs[-1]["speakers"]) | set(item["speakers"]))
            runs[-1]["_probabilities"].append(item.get("probability"))
        else:
            runs.append({"start": item["start"], "end": item["end"], "value": item[key],
                         "unit_count": 1, "speakers": list(item["speakers"]),
                         "_probabilities": [item.get("probability")]})
    for run in runs:
        run["end"] = r3(run["end"])
        run["start"] = r3(run["start"])
        run["duration_s"] = r3(run["end"] - run["start"])
        probabilities = [p for p in run.pop("_probabilities") if p is not None]
        if probabilities:
            run["probability_mean"] = r3(sum(probabilities) / len(probabilities))
            run["probability_min"] = r3(min(probabilities))
            run["probability_max"] = r3(max(probabilities))
    return runs


# ── Phase 3: window-level table and label review diagnostics ────────────────
# Tokens that are statistically common but carry little topical meaning. They
# are *reported*, never removed automatically: the model vocabulary is left
# untouched and a human decides.
GENERIC_LABEL_TERMS = {
    "dont", "don", "doesnt", "didnt", "know", "just", "yeah", "okay", "ok",
    "right", "like", "get", "got", "go", "going", "said", "say", "says",
    "really", "actually", "thing", "things", "time", "times", "people",
    "one", "two", "way", "make", "made", "want", "need", "see", "look",
    "come", "let", "tell", "talk", "going",
}


def intersection_seconds(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float:
    """Seconds of overlap between two closed intervals; 0 if either is invalid."""
    if None in (a_start, a_end, b_start, b_end):
        return 0.0
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def simple_tokens(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def snippet_around(text: str, term: str, before: int = 60, after: int = 140) -> str:
    """A real excerpt of ``text`` centred on ``term`` (never synthesised)."""
    index = text.lower().find(term)
    if index < 0:
        return text[: after + before].strip()
    start = max(0, index - before)
    prefix = "…" if start else ""
    return prefix + text[start:index + len(term) + after].strip()


def build_window_records(
    index: list[dict[str, Any]],
    topics: list[int],
    probabilities: list[float],
    topic_to_subtopic: dict[int, str],
    broad_of_subtopic: dict[str, str],
    episode_segments: dict[str, list[Any]],
    episode_overlaps: dict[str, list[Any]],
    episode_interruptions: dict[str, list[Any]],
    *,
    min_window_seconds: float,
) -> list[dict[str, Any]]:
    """One row per transcript window: topic ids, probability and quality flags."""
    records: list[dict[str, Any]] = []
    for position, meta in enumerate(index):
        video_id = meta["video_id"]
        topic = topics[position]
        subtopic_id = topic_to_subtopic.get(topic, "subtopic_outlier")
        broad_topic_id = broad_of_subtopic.get(subtopic_id, "broad_outlier")
        start, end = finite(meta.get("start")), finite(meta.get("end"))
        duration = (end - start) if (start is not None and end is not None) else None

        segments = episode_segments.get(video_id, [])
        indices = meta.get("source_segment_indices") or []
        logprobs: list[float] = []
        no_speech: list[float] = []
        for seg_index in indices:
            if not isinstance(seg_index, int) or not (0 <= seg_index < len(segments)):
                continue
            segment = segments[seg_index]
            if not isinstance(segment, dict):
                continue
            logprob = finite(segment.get("avg_logprob"))
            speech = finite(segment.get("no_speech_prob"))
            if logprob is not None:
                logprobs.append(logprob)
            if speech is not None:
                no_speech.append(speech)

        flags: list[str] = []
        if not logprobs and not no_speech:
            flags.append("no_confidence_signals")
        if duration is not None and duration < min_window_seconds:
            flags.append("short_window")
        if len(meta.get("speakers") or []) <= 1:
            flags.append("single_speaker")
        if logprobs and (sum(logprobs) / len(logprobs)) < -1.0:
            flags.append("low_mean_logprob")
        if no_speech and max(no_speech) > 0.6:
            flags.append("high_no_speech")

        overlap_hits: list[int] = []
        overlap_seconds = 0.0
        for i, event in enumerate(episode_overlaps.get(video_id, [])):
            if not isinstance(event, dict):
                continue
            seconds = intersection_seconds(start, end, finite(event.get("start")), finite(event.get("end")))
            if seconds > 0:
                overlap_hits.append(i)
                overlap_seconds += seconds
        interruption_hits = [
            i for i, event in enumerate(episode_interruptions.get(video_id, []))
            if isinstance(event, dict)
            and intersection_seconds(start, end, finite(event.get("start")), finite(event.get("end"))) > 0
        ]

        records.append({
            "index": position,
            "video_id": video_id,
            "start": r3(start),
            "end": r3(end),
            "duration_s": r3(duration),
            "speakers": meta.get("speakers") or [],
            "text": meta.get("text"),
            "text_chars": len(meta.get("text") or ""),
            "source_segment_indices": indices,
            "base_topic_id": topic,
            "subtopic_id": subtopic_id,
            "broad_topic_id": broad_topic_id,
            "probability": r3(probabilities[position]),
            "quality": {
                "segment_count": len(indices),
                "mean_avg_logprob": r3(sum(logprobs) / len(logprobs)) if logprobs else None,
                "max_no_speech_prob": r3(max(no_speech)) if no_speech else None,
                "confidence_signals_present": bool(logprobs or no_speech),
                "flags": flags,
            },
            "overlap": {"count": len(overlap_hits), "indices": overlap_hits,
                        "total_seconds": r3(overlap_seconds)},
            "interruption": {"count": len(interruption_hits), "indices": interruption_hits},
            "dropped_reason": None,
        })
    return records


def build_windows_document(
    records: list[dict[str, Any]],
    dropped_regions: list[dict[str, Any]],
    config: TopicConfig,
    model_meta: Optional[dict[str, Any]],
    video_ids: list[str],
) -> dict[str, Any]:
    """Assemble the standalone per-window table artifact."""
    return {
        "schema_version": "topic-windows-1.0",
        "generated_by": "topic_modeling.build_topic_outputs",
        "scope": "corpus_wide",
        "video_ids": video_ids,
        "source_config": asdict(config),
        "model": model_meta,
        "window_count": len(records),
        "windows": records,
        "dropped_regions": dropped_regions,
        "quality_note": ("Per-window quality uses only fields present in the source "
                         "artifacts; Whisper avg_logprob/no_speech_prob are absent here, "
                         "so no window is silently discarded on quality grounds."),
        "limitations": [
            "Windows are model inputs, not observed topic units.",
            "probabilities are BERTopic membership scores, not calibrated confidence.",
            "overlap/interruption hits are time intersections with final.json events.",
        ],
    }


def build_review_document(
    subtopics: list[dict[str, Any]],
    broad_topics: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    records: list[dict[str, Any]],
    embeddings: Any,
    topics: list[int],
) -> dict[str, Any]:
    """Label-quality diagnostics for every candidate subtopic and broad topic.

    Representative examples are chosen as the window nearest each subtopic's
    centroid (not the first member), and every flagged label term is grounded in
    a real transcript excerpt.
    """
    import numpy as np

    window_texts = [record.get("text") or "" for record in records]

    term_subtopics: dict[str, set[str]] = defaultdict(set)
    for subtopic in subtopics:
        for term in subtopic["evidence_terms"]:
            term_subtopics[term].add(subtopic["subtopic_id"])

    candidate_terms = sorted(term_subtopics)
    term_window_counts = {
        term: sum(1 for text in window_texts if term in simple_tokens(text))
        for term in candidate_terms
    }

    stopword_analysis = []
    for term in candidate_terms:
        subtopic_ids = sorted(term_subtopics[term])
        generic = term in GENERIC_LABEL_TERMS or len(subtopic_ids) > 1
        stopword_analysis.append({
            "term": term,
            "subtopic_count": len(subtopic_ids),
            "subtopics": subtopic_ids,
            "window_occurrences": term_window_counts[term],
            "generic": generic,
        })

    subtopic_diagnostics: list[dict[str, Any]] = []
    for subtopic in subtopics:
        base_topic = subtopic["base_topic_id"]
        member_positions = [i for i, topic in enumerate(topics) if topic == base_topic]
        central: Optional[dict[str, Any]] = None
        if member_positions:
            vectors = np.asarray(embeddings)[member_positions]
            centroid = vectors.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            unit = centroid / norm if norm else centroid
            similarities = vectors @ unit
            best = int(np.argmax(similarities))
            best_position = member_positions[best]
            central = {
                "window_index": best_position,
                "start": records[best_position]["start"],
                "end": records[best_position]["end"],
                "centrality_cosine": r3(float(similarities[best])),
                "text": (records[best_position]["text"] or "")[:400],
            }
        grounding = []
        for term in subtopic["evidence_terms"]:
            for position in member_positions:
                if term in simple_tokens(window_texts[position]):
                    grounding.append({
                        "term": term,
                        "window_index": position,
                        "start": records[position]["start"],
                        "excerpt": snippet_around(window_texts[position], term),
                    })
                    break
        subtopic_diagnostics.append({
            "subtopic_id": subtopic["subtopic_id"],
            "broad_topic_id": subtopic["broad_topic_id"],
            "candidate_label": subtopic["candidate_label"],
            "evidence_terms": subtopic["evidence_terms"],
            "unit_count": subtopic["unit_count"],
            "generic_terms": [t for t in subtopic["evidence_terms"]
                              if term_subtopics[t] and (t in GENERIC_LABEL_TERMS or len(term_subtopics[t]) > 1)],
            "central_example": central,
            "first_member_example": subtopic["representative_example"],
            "grounding_examples": grounding,
            "review_status": "candidate",
        })

    subtopic_to_broad = {subtopic["subtopic_id"]: subtopic["broad_topic_id"] for subtopic in subtopics}
    broad_counts = Counter(subtopic_to_broad.values())
    flat_hierarchy = (
        len(broad_topics) == len(subtopics)
        and subtopics
        and all(count == 1 for count in broad_counts.values())
    )

    warnings: list[dict[str, Any]] = []
    if flat_hierarchy:
        warnings.append(issue("flat_hierarchy_no_gain",
                              "Broad topics reproduce subtopics one-to-one; the coarse level adds no grouping."))
    if any(entry["generic"] for entry in stopword_analysis):
        warnings.append(issue("generic_label_terms_present",
                              "Some candidate labels contain non-discriminative tokens; review before use."))

    return {
        "schema_version": "topic-modeling-review-1.0",
        "generated_by": "topic_modeling.build_topic_outputs",
        "central_example_policy": "Representative example = window nearest the subtopic centroid.",
        "subtopics": subtopic_diagnostics,
        "broad_topics": broad_topics,
        "stopword_analysis": stopword_analysis,
        "hierarchy_diagnostics": {
            "broad_topics": len(broad_topics),
            "subtopics": len(subtopics),
            "broad_to_subtopic_counts": {broad["broad_topic_id"]: len(broad["subtopic_ids"]) for broad in broad_topics},
            "subtopic_to_broad_counts": dict(broad_counts),
            "flat_hierarchy": bool(flat_hierarchy),
            "hierarchical_tree_edges": len(hierarchy),
        },
        "warnings": warnings,
        "limitations": [
            "Diagnostics are heuristic and corpus-specific.",
            "Generic-term detection uses a token list plus cross-subtopic repetition.",
            "Centrality uses the same embeddings as the model, so it is not independent evidence.",
        ],
    }


def build_topic_outputs(
    root: Path,
    config: TopicConfig,
    limit: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the pipeline and return (report, windows_document, review_document)."""
    loaded, skipped, failed = load_episodes(root, config.language)
    if limit is not None:
        selected = loaded[:limit]
        selected_ids = {item[0] for item in selected}
        skipped += [{"video_id": video_id, "reason": "not_in_selected_subset"} for video_id, _ in loaded if video_id not in selected_ids]
        loaded = selected

    trim_map = load_trim_map(config.trim_config)
    min_logprob = config.min_avg_logprob if config.drop_unreliable else None
    max_no_speech = config.max_no_speech_prob if config.drop_unreliable else None

    docs: list[str] = []
    index: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    window_quality: dict[str, dict[str, Any]] = {}
    dropped_regions: list[dict[str, Any]] = []
    episode_segments: dict[str, list[Any]] = {}
    episode_overlaps: dict[str, list[Any]] = {}
    episode_interruptions: dict[str, list[Any]] = {}
    for video_id, data in loaded:
        episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
        trim = trim_map.get(video_id, {})
        trim_start_s = float(trim.get("trim_start_s", 0.0))
        units, unit_warnings, unit_stats = build_units(
            data.get("segments"), config.window_seconds, config.min_window_seconds, config.max_window_chars,
            trim_start_s=trim_start_s, min_avg_logprob=min_logprob, max_no_speech_prob=max_no_speech,
        )
        unit_dropped = unit_stats.pop("dropped_ranges", [])
        dropped_regions.extend({**entry, "video_id": video_id} for entry in unit_dropped)
        window_quality[video_id] = {**unit_stats, "trim_start_s": trim_start_s, "trim_reason": trim.get("reason")}
        episode_segments[video_id] = data.get("segments") if isinstance(data.get("segments"), list) else []
        episode_overlaps[video_id] = data.get("overlap_candidates") if isinstance(data.get("overlap_candidates"), list) else []
        episode_interruptions[video_id] = data.get("interruptions") if isinstance(data.get("interruptions"), list) else []
        for warning in unit_warnings:
            warnings.append({**warning, "video_id": video_id})
        for unit in units:
            docs.append(unit["text"])
            index.append({"video_id": video_id, "duration_s": finite(episode.get("duration_s")),
                          "title": episode.get("title"), **unit})

    empty = {"report_metadata": {
        "analysis_version": VERSION, "scope": "corpus_wide", "config": asdict(config),
        "method": "Single BERTopic model over transcript windows; broad topics via agglomerative merge of base topics.",
        "semantic_policy": "All topic labels are candidate vocabulary entries, not observed facts."},
        "episodes": [], "broad_topics": [], "subtopics": [], "hierarchy": [], "topic_vocabulary": {"entries": []},
        "aggregate": {"processing": {"units": 0, "episodes": 0, "analyzed": 0, "skipped": len(skipped), "failed": len(failed)},
                      "dropped": {"ad_trim": 0, "low_logprob": 0, "no_speech": 0, "bad_time": 0, "missing_text": 0},
                      "trimmed_videos": [],
                      "broad_total": 0, "subtopic_total": 0, "outlier_units": 0},
        "skipped": skipped, "failed": failed, "warnings": warnings,
        "limitations": ["English model baseline; non-English artifacts are filtered by default.",
                        "ASR/diarization errors propagate into topic windows.",
                        "Topic labels are model candidates and require human review.",
                        "Topic boundaries and hierarchy are not validated against annotations."],
        "review_template": {"label_overrides": [{"topic_id": "", "approved_name": "", "approved_definition": "", "reviewer": ""}],
                            "broad_merge_overrides": [{"broad_topic_id": "", "subtopic_id": "", "action": "add_or_remove", "rationale": ""}]}}
    video_ids = sorted(video_id for video_id, _ in loaded)
    if len(docs) < config.min_topic_size * 2:
        empty["warnings"].append(issue("insufficient_corpus", "Too few transcript windows for topic modeling", len(docs)))
        empty_windows = build_windows_document([], dropped_regions, config, None, video_ids)
        empty_review = {"schema_version": "topic-modeling-review-1.0",
                        "generated_by": "topic_modeling.build_topic_outputs",
                        "subtopics": [], "broad_topics": [], "stopword_analysis": [],
                        "hierarchy_diagnostics": {"broad_topics": 0, "subtopics": 0, "flat_hierarchy": None,
                                                  "hierarchical_tree_edges": 0},
                        "warnings": empty["warnings"], "limitations": empty["limitations"]}
        return empty, empty_windows, empty_review

    embeddings, model_meta = embed_texts(docs, config.model_name)
    model, topics, probabilities, initial_outliers = fit_model(docs, embeddings, config)
    subtopics, broad_topics, topic_to_subtopic, hierarchy = build_topic_tables(model, docs, topics, embeddings, config)
    broad_of_subtopic = {subtopic["subtopic_id"]: subtopic["broad_topic_id"] for subtopic in subtopics}

    top_info = model.get_topic_info()
    distinct_topics = sorted({topic for topic in topics if topic != -1})
    vocabulary = [{
        "topic_id": subtopic["subtopic_id"], "level": "subtopic",
        "candidate_name": subtopic["candidate_label"], "definition": subtopic["representative_example"],
        "parent_topic_id": subtopic["broad_topic_id"], "unit_count": subtopic["unit_count"], "review_status": "candidate",
    } for subtopic in subtopics] + [{
        "topic_id": broad["broad_topic_id"], "level": "broad", "candidate_name": broad["candidate_label"],
        "definition": None, "parent_topic_id": None, "unit_count": broad["unit_count"], "review_status": "candidate",
    } for broad in broad_topics]

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, meta in enumerate(index):
        topic = topics[position]
        by_episode[meta["video_id"]].append({
            "start": meta["start"], "end": meta["end"], "speakers": meta["speakers"],
            "base_topic_id": topic,
            "subtopic_id": topic_to_subtopic.get(topic, "subtopic_outlier"),
            "broad_topic_id": None,
            "probability": r3(probabilities[position]),
        })
    episodes: list[dict[str, Any]] = []
    for video_id, data in loaded:
        episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
        assignments = by_episode.get(video_id, [])
        for item in assignments:
            item["broad_topic_id"] = broad_of_subtopic.get(item["subtopic_id"], "broad_outlier")
        subtopic_counts = Counter(item["subtopic_id"] for item in assignments)
        broad_counts = Counter(item["broad_topic_id"] for item in assignments)
        episodes.append({
            "metadata": {"video_id": video_id, "title": episode.get("title"),
                         "duration_s": finite(episode.get("duration_s")), "window_count": len(assignments)},
            "window_quality": window_quality.get(video_id, {}),
            "subtopic_distribution": dict(subtopic_counts.most_common()),
            "broad_topic_distribution": dict(broad_counts.most_common()),
            "subtopic_timeline": contiguous_runs(assignments, "subtopic_id"),
            "broad_timeline": contiguous_runs(assignments, "broad_topic_id"),
        })

    records = build_window_records(
        index, topics, probabilities, topic_to_subtopic, broad_of_subtopic,
        episode_segments, episode_overlaps, episode_interruptions,
        min_window_seconds=config.min_window_seconds,
    )
    windows_document = build_windows_document(records, dropped_regions, config, model_meta, video_ids)
    review_document = build_review_document(subtopics, broad_topics, hierarchy, records, embeddings, topics)

    report = {**empty,
              "report_metadata": {**empty["report_metadata"], "model": model_meta},
              "episodes": episodes, "broad_topics": broad_topics, "subtopics": subtopics, "hierarchy": hierarchy,
              "topic_vocabulary": {"entries": vocabulary,
                                   "consolidation_policy": "Labels are candidates; no semantic merge is automatic."},
              "aggregate": {"processing": {"units": len(docs), "episodes": len(episodes), "analyzed": len(episodes),
                                           "skipped": len(skipped), "failed": len(failed)},
                            "dropped": {
                                "ad_trim": sum(q.get("dropped_ad_trim", 0) for q in window_quality.values()),
                                "low_logprob": sum(q.get("dropped_low_logprob", 0) for q in window_quality.values()),
                                "no_speech": sum(q.get("dropped_no_speech", 0) for q in window_quality.values()),
                                "bad_time": sum(q.get("dropped_bad_time", 0) for q in window_quality.values()),
                                "missing_text": sum(q.get("dropped_missing_text", 0) for q in window_quality.values()),
                            },
                            "trimmed_videos": sorted(video_id for video_id, q in window_quality.items() if q.get("trim_start_s")),
                            "broad_total": len(broad_topics), "subtopic_total": len(subtopics),
                            "outlier_units": initial_outliers,
                            "outlier_units_after_reduction": sum(1 for topic in topics if topic == -1),
                            "base_topic_count": len(distinct_topics), "base_topic_info": int(len(top_info))},
              "warnings": warnings + [issue("candidate_labels_require_review", "All topic labels and the hierarchy require human review")]}
    return report, windows_document, review_document


def build_report(root: Path, config: TopicConfig, limit: Optional[int] = None) -> dict[str, Any]:
    """Backwards-compatible wrapper returning only the compact report."""
    report, _, _ = build_topic_outputs(root, config, limit)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-wide hierarchical topic/subtopic modeling from v2 debate artifacts.")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/v2/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/topic_analysis/topic_modeling.json"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--window-seconds", type=float, default=45.0)
    parser.add_argument("--min-topic-size", type=int, default=8)
    parser.add_argument("--broad-topics", type=int, default=8)
    parser.add_argument("--hdbscan-selection", choices=["eom", "leaf"], default="leaf",
                        help="HDBSCAN cluster selection; 'leaf' yields finer subtopics.")
    parser.add_argument("--trim-config", default=None,
                        help="Per-video ad/intro trim map JSON: {\"id\": seconds} or {\"id\": {\"trim_start_s\": s}}.")
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0,
                        help="Drop windows whose Whisper avg_logprob is below this (default: -1.0).")
    parser.add_argument("--max-no-speech-prob", type=float, default=0.6,
                        help="Drop windows whose Whisper no_speech_prob exceeds this (default: 0.6).")
    parser.add_argument("--keep-unreliable", action="store_true",
                        help="Disable the avg_logprob / no_speech_prob quality filter.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N selected episodes (quick runs).")
    parser.add_argument("--windows-output", type=Path, default=Path("data/v2/topic_analysis/topic_windows.json"),
                        help="Per-window table output (text, topic ids, probability, quality/flags).")
    parser.add_argument("--review-output", type=Path, default=Path("data/v2/topic_analysis/topic_modeling_review.json"),
                        help="Label-quality review output (stopword analysis, central examples, hierarchy).")
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.min_topic_size < 2 or args.broad_topics < 1:
        parser.error("window-seconds must be positive, min-topic-size >= 2, broad-topics >= 1.")
    config = TopicConfig(model_name=args.model, language=args.language, window_seconds=args.window_seconds,
                         min_topic_size=args.min_topic_size, n_broad_topics=args.broad_topics,
                         hdbscan_selection=args.hdbscan_selection, trim_config=args.trim_config,
                         drop_unreliable=not args.keep_unreliable, min_avg_logprob=args.min_avg_logprob,
                         max_no_speech_prob=args.max_no_speech_prob)
    report, windows_document, review_document = build_topic_outputs(args.videos_dir, config, args.limit)
    for path, payload in ((args.output, report), (args.windows_output, windows_document),
                          (args.review_output, review_document)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(f"Written {path}")
    status = report["aggregate"]["processing"]
    print(f"Topic modeling: {status['units']} windows, {report['aggregate']['broad_total']} broad topics, "
          f"{report['aggregate']['subtopic_total']} subtopics, {status['failed']} failed")
    print(f"Report saved to {args.output}")
    print(f"Windows saved to {args.windows_output} ({windows_document['window_count']} rows)")
    print(f"Review saved to {args.review_output}")


if __name__ == "__main__":
    main()
