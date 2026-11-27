#!/usr/bin/env python3
"""Hybrid subtopic detection for debate transcripts.

Inputs:
  data/v2/debate_only/videos/<VIDEO_ID>/<VARIANT>/merge_final.json
  data/v2/debate_only/overall_topic/<VARIANT>/<VIDEO_ID>/overall_topic.json

Outputs:
  data/v2/debate_only/subtopic_analysis/<VARIANT>/<VIDEO_ID>/subtopic_analysis.json

Method:
  1. Build overlapping 5-minute transcript chunks (1-minute overlap).
  2. Group transcript segments into short chronological units.
  3. Use multilingual sentence embeddings to propose semantic-change boundaries.
  4. Ask Gemini to verify each candidate using previous/current/following context.
  5. Refine boundaries to actual transcript segment boundaries and merge brief deviations.

The script never modifies source JSON files. It checkpoints verified boundaries so a rerun
can reuse completed LLM decisions. Install requirements listed in the accompanying notes.
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
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "v2" / "debate_only" / "videos"
DEFAULT_OVERALL_DIR = ROOT / "data" / "v2" / "debate_only" / "overall_topic"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "v2" / "debate_only" / "subtopic_analysis"
VARIANTS = ("baseline", "threshold_0.5", "threshold_1.0", "threshold_2.0", "threshold_5.0")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

PROMPT_VERSION = "subtopic-boundary-v2-distinct-issue-angles"

SYSTEM_PROMPT = """You are identifying subtopics in a Hindi/English televised political debate.
The supplied overall topic is the BROAD episode-level subject. Do NOT treat every discussion
under that broad subject as one continuous subtopic.

A NEW SUBTOPIC should be identified when the panel moves to a meaningfully different
historical event, allegation, policy issue, controversy, argument angle, or question that
receives its own discussion. It can remain strongly related to the overall topic and still
be a NEW SUBTOPIC.

Examples of distinct subtopics within one broad debate:
- whether a memoir is a personal account or political whitewashing;
- responsibility for the Emergency;
- the 1984 anti-Sikh riots and political accountability;
- the Babri Masjid demolition and whether it was planned;
- selective grief or comparisons between political parties.
These are distinct subtopics even when all are discussed in relation to the same memoir
or historical accountability. Use the examples only as a guide; follow the supplied transcript.

For each candidate, inspect the previous and following transcript context and ask:
1. Has the central issue or discussion question changed?
2. Is the new issue developed with arguments/evidence, rather than mentioned in passing?
3. Would a researcher reasonably give this section a distinct subtopic title?

Choose exactly one:
- new_subtopic: a distinct issue/discussion angle begins and is developed.
- continuation: the same specific issue or argument angle continues. A speaker change or
  another example within the same issue is not enough.
- brief_deviation: a short aside, interruption, clarification, or temporary detour.
- uncertain: evidence is insufficient or contradictory.

IMPORTANT: The fact that two issues are related to the same overall topic is NOT a reason
to label them continuation. Conversely, a brief mention of another issue is not enough to
create a subtopic. Do not infer facts beyond the transcript. Return JSON only:
{
 "decision": "new_subtopic|continuation|brief_deviation|uncertain",
 "boundary_time": number or null,
 "title": string or null,
 "description": string or null,
 "reason": string,
 "confidence": number between 0 and 1
}
If decision is new_subtopic, boundary_time MUST be the start timestamp of an actual supplied
transcript segment at or immediately around the transition. Use a concise neutral title.
"""

def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj

def num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x >= 0 else None
    except (TypeError, ValueError):
        return None

def extract_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    raw = data.get("segments")
    if not isinstance(raw, list):
        raise ValueError("merge_final.json has no 'segments' array")
    for s in raw:
        if not isinstance(s, dict):
            continue
        start, end, text = num(s.get("start")), num(s.get("end")), s.get("text")
        if start is None or end is None or end < start or not isinstance(text, str) or not text.strip():
            continue
        speaker = s.get("speaker")
        if speaker is None:
            speaker = s.get("speaker_id")
        result.append({
            "start": start, "end": end,
            "speaker_id": speaker if isinstance(speaker, str) else None,
            "text": text.strip()
        })
    result.sort(key=lambda x: (x["start"], x["end"]))
    if not result:
        raise ValueError("No valid transcript segments with text/start/end found")
    return result

def make_chunks(segments: list[dict[str, Any]], chunk_seconds: float, overlap_seconds: float) -> list[dict[str, Any]]:
    last = max(s["end"] for s in segments)
    stride = chunk_seconds - overlap_seconds
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk duration")
    chunks, start, i = [], 0.0, 1
    while start < last:
        end = min(start + chunk_seconds, last)
        members = [s for s in segments if s["start"] < end and s["end"] >= start]
        if members:
            chunks.append({"chunk_id": f"C{i:03d}", "start": start, "end": end, "segments": members})
            i += 1
        start += stride
    return chunks

def build_units(segments: list[dict[str, Any]], unit_seconds: float) -> list[dict[str, Any]]:
    """Build non-overlapping time units for semantic comparison; keep exact source segments."""
    last = max(s["end"] for s in segments)
    units, start, i = [], 0.0, 1
    while start < last:
        end = min(start + unit_seconds, last)
        members = [s for s in segments if start <= s["start"] < end]
        if members:
            units.append({
                "unit_id": f"U{i:04d}", "start": start, "end": end,
                "segments": members,
                "text": " ".join(s["text"] for s in members)
            })
            i += 1
        start = end
    return units

def semantic_candidates(units: list[dict[str, Any]], embedding_model: str,
                        min_gap_seconds: float, similarity_threshold: float,
                        max_candidates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install with: "
            "pip install sentence-transformers"
        ) from exc

    if len(units) < 2:
        return [], {"embedding_model": embedding_model, "units": len(units), "candidate_count": 0}
    model = SentenceTransformer(embedding_model)
    vectors = model.encode([u["text"] for u in units], normalize_embeddings=True, show_progress_bar=False)
    candidates = []
    for i in range(1, len(units)):
        # Cosine similarity for normalized vectors.
        similarity = float((vectors[i - 1] * vectors[i]).sum())
        boundary = units[i]["start"]
        if boundary - (candidates[-1]["time"] if candidates else -1e9) < min_gap_seconds:
            # Keep the strongest candidate in a close cluster.
            if candidates and similarity < candidates[-1]["similarity"]:
                candidates[-1] = {
                    "candidate_id": f"B{len(candidates):04d}",
                    "time": boundary, "similarity": similarity,
                    "semantic_change": 1.0 - similarity,
                    "left_unit_id": units[i - 1]["unit_id"],
                    "right_unit_id": units[i]["unit_id"]
                }
            continue
        if similarity <= similarity_threshold:
            candidates.append({
                "candidate_id": f"B{len(candidates)+1:04d}",
                "time": boundary, "similarity": similarity,
                "semantic_change": 1.0 - similarity,
                "left_unit_id": units[i - 1]["unit_id"],
                "right_unit_id": units[i]["unit_id"]
            })
    # If a transcript is highly repetitive, don't silently fabricate candidates.
    candidates = sorted(candidates, key=lambda x: x["time"])[:max_candidates]
    for i, c in enumerate(candidates, 1):
        c["candidate_id"] = f"B{i:04d}"
    return candidates, {
        "embedding_model": embedding_model,
        "unit_duration_seconds": None,
        "units": len(units),
        "similarity_threshold": similarity_threshold,
        "minimum_candidate_gap_seconds": min_gap_seconds,
        "candidate_count": len(candidates)
    }

def gemini_json(api_key: str, model: str, prompt: str, timeout: int = 120,
                max_attempts: int = 5) -> dict[str, Any]:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    req = urllib.request.Request(
        API_ENDPOINT.format(model=model, key=api_key),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            texts = [
                part["text"]
                for candidate in raw.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
                if isinstance(part.get("text"), str)
            ]
            if not texts:
                raise RuntimeError(f"Gemini returned no text: {raw}")
            result = json.loads("\n".join(texts))
            if not isinstance(result, dict):
                raise ValueError("Gemini response was not a JSON object")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable or attempt == max_attempts - 1:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
            delay = min(15 * (2 ** attempt), 120)
            print(f"Gemini HTTP {exc.code}; retrying in {delay}s ({attempt+2}/{max_attempts})", file=sys.stderr)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Gemini connection failed: {exc}") from exc
            delay = min(15 * (2 ** attempt), 120)
            print(f"Gemini connection error; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("Gemini request failed")

def context_for_candidate(segments: list[dict[str, Any]], t: float, context_seconds: float) -> dict[str, Any]:
    left = [s for s in segments if t - context_seconds <= s["start"] < t]
    right = [s for s in segments if t <= s["start"] <= t + context_seconds]
    # Ensure a small context even when segment timestamps are sparse.
    if not left:
        left = [s for s in segments if s["start"] < t][-8:]
    if not right:
        right = [s for s in segments if s["start"] >= t][:8]
    return {"previous_context": left[-30:], "following_context": right[:30]}

def valid_decision(raw: dict[str, Any], candidate: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    decision = raw.get("decision")
    allowed = {"new_subtopic", "continuation", "brief_deviation", "uncertain"}
    if decision not in allowed:
        decision = "uncertain"
    confidence = num(raw.get("confidence"))
    if confidence is None:
        confidence = 0.0
    confidence = min(1.0, confidence)
    title = raw.get("title") if isinstance(raw.get("title"), str) else None
    description = raw.get("description") if isinstance(raw.get("description"), str) else None
    reason = raw.get("reason") if isinstance(raw.get("reason"), str) else ""
    boundary = num(raw.get("boundary_time"))
    if decision == "new_subtopic":
        # Snap to a real segment start near the model's suggested boundary.
        target = boundary if boundary is not None else candidate["time"]
        nearby = [s["start"] for s in segments if abs(s["start"] - target) <= 20.0]
        if not nearby:
            nearby = [s["start"] for s in segments if abs(s["start"] - candidate["time"]) <= 20.0]
        if not nearby:
            decision = "uncertain"
            boundary = None
        else:
            boundary = min(nearby, key=lambda x: abs(x - target))
            if not title or not title.strip():
                title = "Untitled subtopic"
    else:
        boundary = None
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_time": candidate["time"],
        "semantic_similarity": candidate["similarity"],
        "decision": decision,
        "boundary_time": boundary,
        "title": title.strip() if title else None,
        "description": description.strip() if description else None,
        "reason": reason,
        "confidence": confidence
    }

def evidence_between(segments: list[dict[str, Any]], start: float, end: float, limit: int = 12) -> list[dict[str, Any]]:
    selected = [s for s in segments if s["start"] < end and s["end"] > start]
    if len(selected) > limit:
        # Keep evenly distributed evidence from the interval.
        indices = [round(i * (len(selected)-1) / (limit-1)) for i in range(limit)]
        selected = [selected[i] for i in indices]
    return selected

def make_subtopics(segments: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    duration = max(s["end"] for s in segments)
    starts = sorted({
        float(d["boundary_time"]) for d in decisions
        if d["decision"] == "new_subtopic" and d.get("boundary_time") is not None
    })
    # Avoid zero-length or near-duplicate boundaries.
    clean = []
    for t in starts:
        if 0 < t < duration and (not clean or t - clean[-1] >= 30.0):
            clean.append(t)
    bounds = [0.0] + clean + [duration]
    result = []
    for i, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        if end <= start:
            continue
        # Find nearest accepted LLM decision for a title/description.
        matched = min(
            [d for d in decisions if d["decision"] == "new_subtopic" and d.get("boundary_time") is not None],
            key=lambda d: abs(d["boundary_time"] - start),
            default=None
        ) if start > 0 else None
        segs = [s for s in segments if s["start"] < end and s["end"] > start]
        if not segs:
            continue
        title = matched["title"] if matched and abs(matched["boundary_time"] - start) <= 20 else (
            "Opening discussion" if i == 1 else f"Discussion segment {i}"
        )
        description = matched["description"] if matched and abs(matched["boundary_time"] - start) <= 20 else None
        speakers = sorted({s["speaker_id"] for s in segs if s["speaker_id"]})
        result.append({
            "subtopic_id": f"ST{i:03d}",
            "title": title,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_seconds": round(end-start, 3),
            "description": description,
            "speakers": speakers,
            "evidence": evidence_between(segments, start, end)
        })
    return result

def run_one(video_id: str, variant: str, transcript_path: Path, overall_path: Path,
            output_path: Path, checkpoint_path: Path, args: argparse.Namespace,
            api_key: str) -> tuple[int, int]:
    transcript = load_json(transcript_path)
    overall_data = load_json(overall_path)
    overall = overall_data.get("overall_topic", {})
    if not isinstance(overall, dict) or not isinstance(overall.get("topic"), str):
        raise ValueError(f"Invalid overall_topic.json: {overall_path}")
    segments = extract_segments(transcript)
    chunks = make_chunks(segments, args.chunk_minutes * 60, args.overlap_minutes * 60)
    units = build_units(segments, args.unit_seconds)
    candidates, embedding_config = semantic_candidates(
        units, args.embedding_model, args.min_candidate_gap,
        args.similarity_threshold, args.max_candidates
    )

    checkpoint = {
        "video_id": video_id, "variant": variant,
        "prompt_version": PROMPT_VERSION, "decisions": []
    }
    if checkpoint_path.exists():
        try:
            saved = load_json(checkpoint_path)
            if (saved.get("video_id") == video_id
                    and saved.get("variant") == variant
                    and saved.get("prompt_version") == PROMPT_VERSION):
                checkpoint = saved
        except Exception:
            pass
    # Reuse a checkpoint decision only when its candidate timestamp still matches.
    saved_by_id = {
        d.get("candidate_id"): d for d in checkpoint.get("decisions", [])
        if isinstance(d, dict)
    }
    decisions = []
    for candidate in candidates:
        saved_decision = saved_by_id.get(candidate["candidate_id"])
        if (saved_decision is not None
                and abs(float(saved_decision.get("candidate_time", -9999))
                        - candidate["time"]) < 0.001):
            decisions.append(saved_decision)
            continue
        context = context_for_candidate(segments, candidate["time"], args.context_seconds)
        prompt_payload = {
            "overall_topic": overall.get("topic"),
            "overall_topic_description": overall.get("description"),
            "candidate_boundary": candidate,
            **context,
            "instruction": "Decide whether this candidate begins a distinct subtopic. Brief deviations should be merged."
        }
        raw = gemini_json(api_key, args.model, json.dumps(prompt_payload, ensure_ascii=False))
        decision = valid_decision(raw, candidate, segments)
        decisions.append(decision)
        checkpoint["prompt_version"] = PROMPT_VERSION
        checkpoint["decisions"] = decisions
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subtopics = make_subtopics(segments, decisions)
    result = {
        "schema_version": "subtopic-analysis-1.0",
        "video_id": video_id,
        "variant": variant,
        "source_files": {
            "transcript": str(transcript_path),
            "overall_topic": str(overall_path)
        },
        "configuration": {
            "method": "hybrid_semantic_similarity_and_llm_verification",
            "prompt_version": PROMPT_VERSION,
            "chunk_duration_minutes": args.chunk_minutes,
            "overlap_minutes": args.overlap_minutes,
            "semantic_unit_seconds": args.unit_seconds,
            "embedding_model": args.embedding_model,
            "semantic_similarity_threshold": args.similarity_threshold,
            "gemini_model": args.model,
            "brief_deviations": "merged_into_surrounding_subtopic"
        },
        "overall_topic": {
            "topic": overall.get("topic"),
            "description": overall.get("description")
        },
        "chunks": [{"chunk_id": c["chunk_id"], "start": c["start"], "end": c["end"],
                    "segment_count": len(c["segments"])} for c in chunks],
        "boundary_analysis": {
            "embedding": embedding_config,
            "candidates": decisions,
            "counts": dict(Counter(d["decision"] for d in decisions))
        },
        "subtopics": subtopics,
        "summary": {
            "total_subtopics": len(subtopics),
            "total_duration_seconds": round(max(s["end"] for s in segments), 3),
            "candidate_boundaries": len(candidates),
            "verified_boundaries": sum(d["decision"] == "new_subtopic" for d in decisions)
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return len(chunks), len(candidates)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--overall-topic-dir", type=Path, default=DEFAULT_OVERALL_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--video-id")
    p.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--chunk-minutes", type=float, default=5.0)
    p.add_argument("--overlap-minutes", type=float, default=1.0)
    p.add_argument("--unit-seconds", type=float, default=30.0)
    p.add_argument("--similarity-threshold", type=float, default=0.55,
                   help="Create candidate when adjacent-unit cosine similarity is at or below this value.")
    p.add_argument("--min-candidate-gap", type=float, default=45.0)
    p.add_argument("--context-seconds", type=float, default=90.0)
    p.add_argument("--max-candidates", type=int, default=100)
    p.add_argument("--force", action="store_true", help="Recompute even if final output exists.")
    args = p.parse_args()
    if args.chunk_minutes <= 0 or args.overlap_minutes < 0 or args.overlap_minutes >= args.chunk_minutes:
        p.error("Require chunk-minutes > 0 and 0 <= overlap-minutes < chunk-minutes")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.data_dir.is_dir():
        print(f"ERROR: data directory not found: {args.data_dir}", file=sys.stderr)
        return 2
    variants = (args.variant,) if args.variant else VARIANTS
    video_dirs = [args.data_dir / args.video_id] if args.video_id else sorted(x for x in args.data_dir.iterdir() if x.is_dir())
    totals = Counter(videos_discovered=len(video_dirs), variants_discovered=len(variants))
    for vd in video_dirs:
        if not vd.is_dir():
            print(f"[WARNING] video directory missing: {vd}", file=sys.stderr)
            continue
        for variant in variants:
            transcript_path = vd / variant / "merge_final.json"
            overall_path = args.overall_topic_dir / variant / vd.name / "overall_topic.json"
            output_path = args.output_dir / variant / vd.name / "subtopic_analysis.json"
            checkpoint_path = args.output_dir / "_checkpoints" / variant / vd.name / "boundary_checkpoint.json"
            if not transcript_path.is_file():
                totals["missing_transcript_files"] += 1
                print(f"[WARNING] missing transcript: {transcript_path}", file=sys.stderr)
                continue
            if not overall_path.is_file():
                totals["missing_overall_topic_files"] += 1
                print(f"[WARNING] missing overall topic: {overall_path}", file=sys.stderr)
                continue
            if output_path.exists() and not args.force:
                try:
                    existing = load_json(output_path)
                    if existing.get("video_id") == vd.name and existing.get("variant") == variant:
                        totals["skipped_existing"] += 1
                        print(f"[SKIP] {vd.name} / {variant} (output exists)")
                        continue
                except Exception:
                    pass
            totals["jobs_attempted"] += 1
            try:
                n_chunks, n_candidates = run_one(vd.name, variant, transcript_path, overall_path,
                                                 output_path, checkpoint_path, args, api_key)
                totals["successful_jobs"] += 1
                totals["chunks_processed"] += n_chunks
                totals["candidate_boundaries"] += n_candidates
                totals["subtopics"] += len(load_json(output_path).get("subtopics", []))
                print(f"[OK] {vd.name} / {variant}: {n_chunks} chunks, {n_candidates} candidates")
            except Exception as exc:
                totals["failed_jobs"] += 1
                print(f"[ERROR] {vd.name} / {variant}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("\nSubtopic detection summary")
    for key in ("videos_discovered", "variants_discovered", "jobs_attempted", "successful_jobs",
                "failed_jobs", "skipped_existing", "missing_transcript_files", "missing_overall_topic_files",
                "chunks_processed", "candidate_boundaries", "subtopics"):
        print(f"  {key.replace('_', ' ')}: {totals[key]}")
    print(f"  output directory: {args.output_dir}")
    return 1 if totals["failed_jobs"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
