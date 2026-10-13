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


def build_units(segments: list[Any], window_s: float, min_window_s: float, max_chars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge consecutive timed segments into windows of roughly window_s seconds.

    Single debate turns are too short for stable topical signal; windows give the
    embedding model enough lexical context.
    """
    windows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    bad = 0
    buffer: list[dict[str, Any]] = []

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
        if not isinstance(segment, dict):
            bad += 1
            continue
        text = segment.get("text")
        start, end = finite(segment.get("start")), finite(segment.get("end"))
        if not isinstance(text, str) or not text.strip():
            warnings.append(issue("missing_transcript_text", "A segment lacks transcript text"))
            continue
        if start is None or end is None or end <= start:
            bad += 1
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
    if bad:
        warnings.append(issue("invalid_segment_time_excluded", "Segment has invalid timestamp", bad))
    return windows, warnings


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
        else:
            runs.append({"start": item["start"], "end": item["end"], "value": item[key],
                         "unit_count": 1, "speakers": list(item["speakers"])})
    for run in runs:
        run["end"] = r3(run["end"])
        run["start"] = r3(run["start"])
        run["duration_s"] = r3(run["end"] - run["start"])
    return runs


def build_report(root: Path, config: TopicConfig, limit: Optional[int] = None) -> dict[str, Any]:
    loaded, skipped, failed = load_episodes(root, config.language)
    if limit is not None:
        selected = loaded[:limit]
        selected_ids = {item[0] for item in selected}
        skipped += [{"video_id": video_id, "reason": "not_in_selected_subset"} for video_id, _ in loaded if video_id not in selected_ids]
        loaded = selected

    docs: list[str] = []
    index: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for video_id, data in loaded:
        episode = data.get("episode") if isinstance(data.get("episode"), dict) else {}
        units, unit_warnings = build_units(data.get("segments"), config.window_seconds, config.min_window_seconds, config.max_window_chars)
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
                      "broad_total": 0, "subtopic_total": 0, "outlier_units": 0},
        "skipped": skipped, "failed": failed, "warnings": warnings,
        "limitations": ["English model baseline; non-English artifacts are filtered by default.",
                        "ASR/diarization errors propagate into topic windows.",
                        "Topic labels are model candidates and require human review.",
                        "Topic boundaries and hierarchy are not validated against annotations."],
        "review_template": {"label_overrides": [{"topic_id": "", "approved_name": "", "approved_definition": "", "reviewer": ""}],
                            "broad_merge_overrides": [{"broad_topic_id": "", "subtopic_id": "", "action": "add_or_remove", "rationale": ""}]}}
    if len(docs) < config.min_topic_size * 2:
        empty["warnings"].append(issue("insufficient_corpus", "Too few transcript windows for topic modeling", len(docs)))
        return empty

    embeddings, model_meta = embed_texts(docs, config.model_name)
    model, topics, probabilities, initial_outliers = fit_model(docs, embeddings, config)
    subtopics, broad_topics, topic_to_subtopic, hierarchy = build_topic_tables(model, docs, topics, embeddings, config)

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
    broad_of_subtopic = {subtopic["subtopic_id"]: subtopic["broad_topic_id"] for subtopic in subtopics}
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
            "subtopic_distribution": dict(subtopic_counts.most_common()),
            "broad_topic_distribution": dict(broad_counts.most_common()),
            "subtopic_timeline": contiguous_runs(assignments, "subtopic_id"),
            "broad_timeline": contiguous_runs(assignments, "broad_topic_id"),
        })

    return {**empty,
            "report_metadata": {**empty["report_metadata"], "model": model_meta},
            "episodes": episodes, "broad_topics": broad_topics, "subtopics": subtopics, "hierarchy": hierarchy,
            "topic_vocabulary": {"entries": vocabulary,
                                 "consolidation_policy": "Labels are candidates; no semantic merge is automatic."},
            "aggregate": {"processing": {"units": len(docs), "episodes": len(episodes), "analyzed": len(episodes),
                                         "skipped": len(skipped), "failed": len(failed)},
                          "broad_total": len(broad_topics), "subtopic_total": len(subtopics),
                          "outlier_units": initial_outliers,
                          "outlier_units_after_reduction": sum(1 for topic in topics if topic == -1),
                          "base_topic_count": len(distinct_topics), "base_topic_info": int(len(top_info))},
            "warnings": warnings + [issue("candidate_labels_require_review", "All topic labels and the hierarchy require human review")]}


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
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N selected episodes (quick runs).")
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.min_topic_size < 2 or args.broad_topics < 1:
        parser.error("window-seconds must be positive, min-topic-size >= 2, broad-topics >= 1.")
    config = TopicConfig(model_name=args.model, language=args.language, window_seconds=args.window_seconds,
                         min_topic_size=args.min_topic_size, n_broad_topics=args.broad_topics,
                         hdbscan_selection=args.hdbscan_selection)
    report = build_report(args.videos_dir, config, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    status = report["aggregate"]["processing"]
    print(f"Topic modeling: {status['units']} windows, {report['aggregate']['broad_total']} broad topics, "
          f"{report['aggregate']['subtopic_total']} subtopics, {status['failed']} failed")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
