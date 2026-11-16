#!/usr/bin/env python3
"""Create a compact, source-traceable five-stage comparison report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


STAGES = (
    ("original", "Baseline (without merge)", "."),
    ("baseline", "Baseline (with merge)", "baseline"),
    ("threshold_0.5", "Threshold 0.5 s", "threshold_0.5"),
    ("threshold_1.0", "Threshold 1.0 s", "threshold_1.0"),
    ("threshold_2.0", "Threshold 2.0 s", "threshold_2.0"),
)


def read(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def turns(data):
    rows = []
    for row in (data or {}).get("speaker_turns", []):
        try:
            speaker, start, end = row["speaker"], float(row["start"]), float(row["end"])
            if speaker and end >= start:
                rows.append((str(speaker), start, end))
        except (KeyError, TypeError, ValueError):
            pass
    return sorted(rows, key=lambda row: (row[1], row[2]))


def coverage(intervals):
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return round(sum(end - start for start, end in merged), 3)


def overlaps(data):
    records = (data or {}).get("overlap_candidates", [])
    episodes, actual = [], []
    for record in records if isinstance(records, list) else []:
        try:
            start, end = float(record["start"]), float(record["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        episodes.append((start, end))
        sources = record.get("source_regions") if isinstance(record, dict) else None
        for source in sources if isinstance(sources, list) else [record]:
            try:
                actual.append((float(source["start"]), float(source["end"])))
            except (KeyError, TypeError, ValueError):
                pass
    return len(episodes), coverage(actual)


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def duration_text(seconds):
    if seconds is None:
        return None
    seconds = round(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def stage(root: Path, key: str, label: str, rel: str):
    directory = root / rel
    final = read(directory / "final.json")
    source = turns(final)
    merged_data = read(directory / "merge_final.json") if rel != "." else None
    merged = turns(merged_data) if merged_data else source
    gap = read(directory / "turn_gap_analysis.json")
    debate = ((gap or {}).get("statistics") or {}).get("debate_level") or {}
    shares, total = {}, sum(end - start for _, start, end in source)
    for speaker, start, end in source:
        shares[speaker] = shares.get(speaker, 0) + end - start
    shares = {speaker: round(100 * value / total, 3) if total else None for speaker, value in shares.items()}
    transitions = sum(a[0] != b[0] for a, b in zip(source, source[1:]))
    events, actual_duration = overlaps(final)
    duration = None
    try:
        duration = float(final["episode"].get("duration_s", final["episode"].get("duration_seconds")))
    except (KeyError, TypeError, ValueError):
        pass
    gap_total = ((debate.get("total_gap_statistics") or {}).get("total")) if gap else None
    return {"stage": key, "label": label,
            "sources": {"final": str(directory / "final.json"), "merged": str(directory / "merge_final.json") if merged_data else None, "gap": str(directory / "turn_gap_analysis.json") if gap else None},
            "metrics": {"video_duration_seconds": duration, "video_duration": duration_text(duration), "speaker_count": len({x[0] for x in source}) if final else None,
                        "structural_speaker_transitions": transitions if final else None, "overlap_events": events if final else None,
                        "actual_overlap_duration_seconds": actual_duration if final else None, "original_speaker_turns": len(source) if final else None,
                        "merged_speaker_turns": len(merged) if final else None,
                        "merge_operations": len(source) - len(merged) if final else None,
                        "total_gap_duration_seconds": gap_total}, "speaker_shares_percent": shares}


def table(rows, stages):
    head = "<tr><th>Metric</th>" + "".join(f"<th>{html.escape(s['label'])}</th>" for s in stages) + "</tr>"
    body = "".join("<tr><th>" + label + "</th>" + "".join(f"<td>{fmt(s['metrics'][key])}</td>" for s in stages) + "</tr>" for key, label in rows)
    return f"<table>{head}{body}</table>"


def bars(title, key, stages):
    values = [s["metrics"].get(key) or 0 for s in stages]; maximum = max(values) or 1
    rows = "".join(f"<div class='barrow'><span>{html.escape(s['label'])}</span><i style='width:{100*v/maximum:.1f}%'></i><b>{fmt(v)}</b></div>" for s, v in zip(stages, values))
    return f"<section><h2>{title}</h2>{rows}</section>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2/debate_only/videos"))
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args(); root = args.data_dir / args.video_id
    stages = [stage(root, *item) for item in STAGES]
    speakers = sorted({speaker for item in stages for speaker in item["speaker_shares_percent"]})
    report = {"schema_version": "five-stage-compact-1.0", "video_id": args.video_id, "metric_definitions": {"overlap_duration": "union of original constituent overlap intervals; not merged-episode span", "merge_operations": "original turn count minus merged analytical turn count", "speaker_transitions": "speaker-ID changes in ordered original turns", "speaker_shares": "actual attributed turn duration / total attributed turn duration"}, "stages": stages}
    out = root / "comparison"; out.mkdir(parents=True, exist_ok=True)
    (out / "compact_five_stage_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = (("video_duration", "Video duration (MM:SS)"), ("speaker_count", "Speaker count"), ("structural_speaker_transitions", "Speaker transitions"), ("overlap_events", "Overlap events"), ("actual_overlap_duration_seconds", "Actual overlap duration (s)"), ("original_speaker_turns", "Original speaker turns"), ("merged_speaker_turns", "Merged speaker turns"), ("merge_operations", "Merge operations"), ("total_gap_duration_seconds", "Total gap duration (s)"))
    speaker_table = "<table><tr><th>Speaker</th>" + "".join(f"<th>{html.escape(s['label'])}</th>" for s in stages) + "</tr>" + "".join("<tr><th>"+html.escape(sp)+"</th>"+"".join(f"<td>{fmt(s['speaker_shares_percent'].get(sp))}</td>" for s in stages)+"</tr>" for sp in speakers) + "</table>"
    page = "<html><head><meta charset='utf-8'><title>Five-stage comparison</title><style>body{font:14px system-ui;margin:28px;color:#172033}h1{margin:0}h2{margin-top:28px}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{padding:7px;border-bottom:1px solid #d9dee8;text-align:right}th:first-child{text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.barrow{display:grid;grid-template-columns:150px 1fr 55px;gap:8px;margin:8px 0;align-items:center}.barrow i{height:16px;background:#2563eb;display:block;border-radius:3px}.note{color:#526074}@media(max-width:800px){.grid{grid-template-columns:1fr}.barrow{grid-template-columns:120px 1fr 50px}}</style></head><body><h1>Five-stage video analysis comparison</h1><p class='note'>Overlap duration is actual interval coverage; merged episode span is not substituted. Thresholds retain episodes whose merged span meets the threshold.</p>" + table(metrics, stages) + "<h2>Speaker shares (% actual attributed speech)</h2>" + speaker_table + "<div class='grid'>" + bars("Overlap events", "overlap_events", stages) + bars("Actual overlap duration (seconds)", "actual_overlap_duration_seconds", stages) + bars("Original speaker turns", "original_speaker_turns", stages) + bars("Merged speaker turns", "merged_speaker_turns", stages) + "</div></body></html>"
    (out / "compact_five_stage_comparison.html").write_text(page, encoding="utf-8")
    print(table(metrics, stages).replace("<tr>", "\n").replace("</tr>", "").replace("<th>", " | ").replace("</th>", "").replace("<td>", " | ").replace("</td>", "").replace("<table>", "").replace("</table>", ""))
    print(out / "compact_five_stage_comparison.json"); print(out / "compact_five_stage_comparison.html")


if __name__ == "__main__": main()
