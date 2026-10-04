import json
from collections import defaultdict, Counter
from pathlib import Path

VIDEOS_DIR = Path("data/v2/videos")
OUT_DIR = Path("data/v2/analysis")

MERGE_GAP_S = 1.0


def load_episodes():
    episodes = []
    skipped = []
    for video_dir in sorted(VIDEOS_DIR.iterdir()):
        final = video_dir / "final.json"
        if not final.exists():
            skipped.append(video_dir.name)
            continue
        with open(final) as f:
            episodes.append(json.load(f))
    return episodes, skipped


def merge_turns(turns, gap_s=MERGE_GAP_S):
    merged = []
    for t in turns:
        if merged and t["speaker"] == merged[-1]["speaker"] and t["start"] - merged[-1]["end"] <= gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append({"speaker": t["speaker"], "start": t["start"], "end": t["end"]})
    return [t for t in merged if t["end"] > t["start"]]


def participation(turns):
    per_speaker = defaultdict(float)
    turn_counts = Counter()
    for t in turns:
        per_speaker[t["speaker"]] += t["end"] - t["start"]
        turn_counts[t["speaker"]] += 1
    total = sum(per_speaker.values())
    return {
        "speaking_s": dict(sorted(per_speaker.items(), key=lambda kv: -kv[1])),
        "turn_counts": dict(turn_counts.most_common()),
        "total_speech_s": round(total, 2),
        "share_pct": {k: round(100 * v / total, 1) for k, v in per_speaker.items()} if total else {},
    }


def transitions(turns):
    matrix = defaultdict(Counter)
    seq = []
    for t in turns:
        if not seq or t["speaker"] != seq[-1]:
            seq.append(t["speaker"])
    for a, b in zip(seq, seq[1:]):
        matrix[a][b] += 1
    return {a: dict(c.most_common()) for a, c in matrix.items()}, seq


def interruption_table(interruptions):
    table = Counter()
    for it in interruptions:
        table[(it["interrupter"], it["speakers"][0] if it["speakers"][0] != it["interrupter"] else it["speakers"][1])] += 1
    return [{"interrupter": k[0], "victim": k[1], "count": v} for k, v in table.most_common()]


def host_from_roster(ep):
    for p in ep.get("participants", []):
        if p.get("role") == "anchor":
            return {"speaker": p["speaker_label"], "name": p.get("name"), "basis": "roster"}
    return None


def infer_host(turns, interruptions):
    speech = defaultdict(float)
    turn_counts = Counter()
    for t in turns:
        speech[t["speaker"]] += t["end"] - t["start"]
        turn_counts[t["speaker"]] += 1
    total = sum(speech.values())
    eligible = {s for s, v in speech.items() if total and v / total >= 0.01}
    if not eligible:
        return None
    out_degree = Counter()
    for it in interruptions:
        out_degree[it["interrupter"]] += 1
    max_turns = max(turn_counts.values())
    max_out = max(out_degree.values()) if out_degree else 0
    scores = defaultdict(float)
    for s in eligible:
        scores[s] = 0.0
        if turn_counts[s] == max_turns:
            scores[s] += 1
        if out_degree and out_degree[s] == max_out and max_out > 0:
            scores[s] += 1
    first = turns[0]["speaker"]
    if first in scores:
        scores[first] += 2
    host = max(sorted(scores), key=lambda s: scores[s])
    top = scores[host]
    tied = [s for s, v in scores.items() if v == top]
    return {
        "speaker": host,
        "name": None,
        "basis": "inferred",
        "signals": {"first_speaker": first == host, "most_turns": turn_counts[host] == max_turns, "most_interruptions_initiated": bool(out_degree) and out_degree[host] == max_out},
        "tied_with": tied if len(tied) > 1 else [],
    }


def resolve_host(ep, turns):
    roster = host_from_roster(ep)
    if roster:
        return roster
    return infer_host(merge_turns(ep.get("speaker_turns", [])), ep.get("interruptions", []))


def quality_flags(ep, turns):
    speakers = sorted({t["speaker"] for t in turns})
    bad_turns = sum(1 for t in turns if t["end"] - t["start"] <= 0)
    flags = []
    if len(speakers) > 8:
        flags.append(f"many_speakers({len(speakers)})")
    if not ep.get("participants"):
        flags.append("no_participants")
    if bad_turns:
        flags.append(f"nonpositive_turns({bad_turns})")
    osd = ep.get("osd_stats", {})
    if osd.get("num_interruptions", 0) > 0 and osd.get("overlap_seconds", 0) <= 0:
        flags.append("interr_without_overlap")
    return {"n_speakers": len(speakers), "flags": flags}


def per_minute(x, duration_s):
    m = duration_s / 60.0
    return round(x / m, 2) if m > 0 else None


def analyze_episode(ep):
    turns = merge_turns(ep.get("speaker_turns", []))
    part = participation(turns)
    trans, seq = transitions(turns)
    osd = ep.get("osd_stats", {})
    duration = ep.get("episode", {}).get("duration_s") or osd.get("total_duration_s") or 0
    interr = ep.get("interruptions", [])
    host = resolve_host(ep, turns)
    top_trans = sorted(
        ((a, b, n) for a, bs in trans.items() for b, n in bs.items()),
        key=lambda x: -x[2],
    )[:5]
    return {
        "video_id": ep["episode"]["video_id"],
        "title": ep["episode"]["title"],
        "channel": ep["episode"].get("channel"),
        "duration_min": round(duration / 60, 1),
        "n_raw_turns": len(ep.get("speaker_turns", [])),
        "n_merged_turns": len(turns),
        "participation": part,
        "transitions": trans,
        "top_transitions": [f"{a}->{b}: {n}" for a, b, n in top_trans],
        "n_sequence_switches": sum(1 for a, b in zip(seq, seq[1:]) if a != b),
        "host": host,
        "overlap_seconds": osd.get("overlap_seconds"),
        "overlap_pct_of_duration": round(100 * osd["overlap_seconds"] / duration, 2) if duration and osd.get("overlap_seconds") is not None else None,
        "n_overlaps": len(ep.get("overlap_candidates", [])),
        "n_interruptions": len(interr),
        "interruptions_per_min": per_minute(len(interr), duration),
        "overlap_regions_per_min": per_minute(osd.get("num_regions", 0), duration),
        "interruptions_by_pair": interruption_table(interr),
        "quality": quality_flags(ep, turns),
    }


def aggregate(results):
    agg = {
        "n_episodes": len(results),
        "duration_min_range": [min(r["duration_min"] for r in results), max(r["duration_min"] for r in results)],
        "n_speakers_range": [min(r["quality"]["n_speakers"] for r in results), max(r["quality"]["n_speakers"] for r in results)],
        "interruptions_per_min": {
            "min": min(r["interruptions_per_min"] or 0 for r in results),
            "max": max(r["interruptions_per_min"] or 0 for r in results),
            "median": sorted(r["interruptions_per_min"] or 0 for r in results)[len(results) // 2],
        },
        "overlap_pct": {
            "min": min(r["overlap_pct_of_duration"] or 0 for r in results),
            "max": max(r["overlap_pct_of_duration"] or 0 for r in results),
        },
        "episodes_with_participants": sum(1 for r in results if "no_participants" not in r["quality"]["flags"]),
        "episodes_flagged": sum(1 for r in results if r["quality"]["flags"]),
    }
    return agg


def print_report(results, agg, skipped):
    print(f"Episodes analyzed: {len(results)} | skipped (no final.json): {len(skipped)}")
    if skipped:
        print("Skipped:", ", ".join(skipped))
    print()
    header = f"{'video_id':<12} {'dur(m)':>6} {'spk':>3} {'host':<11} {'basis':<8} {'turns':>6} {'speech_s':>8} {'ovl%':>5} {'intr':>4} {'intr/m':>6} flags"
    print(header)
    print("-" * len(header))
    for r in results:
        q = r["quality"]
        host = r.get("host") or {}
        print(
            f"{r['video_id']:<12} {r['duration_min']:>6.1f} {q['n_speakers']:>3} "
            f"{host.get('speaker', '-'):<11} {host.get('basis', '-'):<8} "
            f"{r['n_merged_turns']:>6} {r['participation']['total_speech_s']:>8.0f} "
            f"{(r['overlap_pct_of_duration'] or 0):>5.2f} {r['n_interruptions']:>4} "
            f"{(r['interruptions_per_min'] or 0):>6.2f} {'; '.join(q['flags']) or '-'}"
        )
    print()
    print("Aggregate:", json.dumps(agg, indent=2))
    print()
    for r in results:
        if not r["quality"]["flags"]:
            continue
        print(f"--- {r['video_id']} (flags: {'; '.join(r['quality']['flags'])})")
        print(f"    title: {r['title']}")
        print(f"    speaking share: {r['participation']['share_pct']}")
        print(f"    top transitions: {', '.join(r['top_transitions'])}")
        print(f"    interruptions: {r['interruptions_by_pair']}")
        print()


def main():
    episodes, skipped = load_episodes()
    results = [analyze_episode(ep) for ep in episodes]
    agg = aggregate(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"aggregate": agg, "episodes": results, "skipped": skipped}
    with open(OUT_DIR / "discourse_summary.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(results, agg, skipped)
    print(f"Report saved to {OUT_DIR / 'discourse_summary.json'}")


if __name__ == "__main__":
    main()
