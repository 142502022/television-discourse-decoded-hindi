#!/usr/bin/env python3
"""Build a self-contained, threshold-controlled multi-video research dashboard."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = {
    "baseline": "merged_no_threshold",
    "0.5": "threshold_0.5",
    "1.0": "threshold_1.0",
    "2.0": "threshold_2.0",
    "5.0": "threshold_5.0",
}


def load(path: Path, issues: list[dict], label: str) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        raise ValueError("expected JSON object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append({"source": label, "path": str(path), "reason": str(exc)})
        return None


def number(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value == value else None
    except (TypeError, ValueError):
        return None


def host_turn_stats(final: dict | None, host_id: str | None) -> dict:
    values = []
    for row in (final or {}).get("speaker_turns", []):
        if not isinstance(row, dict) or row.get("speaker") != host_id:
            continue
        start, end = number(row.get("start")), number(row.get("end"))
        if start is not None and end is not None and end > start:
            values.append(end - start)
    return {"longest_seconds": round(max(values), 3) if values else None,
            "mean_seconds": round(sum(values) / len(values), 3) if values else None,
            "turn_count": len(values)}


def build_video(video_dir: Path, issues: list[dict]) -> dict:
    comparison = load(video_dir / "comparison" / "five_stage_comparison.json", issues, "comparison")
    if not comparison:
        return {"video_id": video_dir.name, "stages": {}, "available": False}
    result = {"video_id": video_dir.name, "available": True, "stages": {}, "overview": {}}
    raw_stages = comparison.get("stage_results", {})
    source = raw_stages.get("original", {})
    result["overview"]["duration_seconds"] = source.get("timeline_statistics", {}).get("episode_duration_seconds")
    ads = load(video_dir / "advertisement_analysis" / "advertisement_exclusions.json", [], "advertisements")
    # None is intentional when no removal record is available: never convert it to zero.
    result["overview"]["ads_removed"] = ((ads or {}).get("summary", {}).get("total_excluded_advertisements")
                                            if ads else None)
    for selector, stage_name in STAGES.items():
        stage = raw_stages.get(stage_name)
        if not isinstance(stage, dict):
            result["stages"][selector] = {"available": False}
            continue
        participation = stage.get("speaker_participation", {})
        by_speaker = participation.get("by_speaker", {}) if isinstance(participation, dict) else {}
        host_id = stage.get("host_analysis", {}).get("identified_host_speaker")
        final = load(video_dir / ("baseline" if selector == "baseline" else f"threshold_{selector}") / "final.json", [], "stage final")
        gap = stage.get("gap_analysis", {})
        overlap = stage.get("overlap_statistics", {})
        host = by_speaker.get(host_id, {}) if host_id else {}
        result["stages"][selector] = {
            "available": True,
            "speaker_count": len(by_speaker) if by_speaker else None,
            "speaker_shares": {speaker: {"original_percent": row.get("speech_share_percent_of_attributed_total"),
                                            "clean_percent": row.get("clean_speech_share_percent"),
                                            "attributed_seconds": row.get("attributed_speaking_seconds"),
                                            "clean_seconds": row.get("clean_speech_seconds"),
                                            "excluded_seconds": row.get("speech_excluded_by_retained_overlap_seconds")}
                               for speaker, row in by_speaker.items()},
            "host": {"speaker_id": host_id, "original_share_percent": host.get("speech_share_percent_of_attributed_total"),
                     "clean_share_percent": host.get("clean_speech_share_percent"), **host_turn_stats(final, host_id)},
            "overlap_events": overlap.get("total_overlap_events"),
            "actual_overlap_duration_seconds": overlap.get("duration", {}).get("total"),
            "clean_silence_duration_seconds": gap.get("clean_silence_duration"),
            "total_inter_turn_gap_duration_seconds": gap.get("total_inter_turn_gap_duration", {}).get("total"),
            "gaps_containing_overlap_duration_seconds": gap.get("gaps_containing_overlap_duration"),
            "clean_silence_gap_count": gap.get("clean_silence_gap_count"),
            "gaps_containing_overlap_count": gap.get("gaps_containing_overlap_count"),
        }
    return result


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Hindi TV Debate Research Dashboard</title><style>
body{font:14px system-ui,sans-serif;margin:0;color:#172033;background:#f6f8fb}main{max-width:1500px;margin:auto;padding:28px}.controls,.card{background:white;border:1px solid #dce3ee;border-radius:10px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}.wide{grid-column:1/-1}.bar{display:flex;align-items:end;gap:7px;height:230px;border-bottom:1px solid #9aa7b8;padding:12px 5px}.item{flex:1;min-width:28px;height:100%;display:flex;align-items:end;justify-content:center;position:relative}.item i{width:75%;background:#2463a5;border-radius:4px 4px 0 0}.item small{position:absolute;bottom:-36px;writing-mode:vertical-rl;transform:rotate(180deg);max-height:70px;overflow:hidden}.stack{display:flex;height:42px;margin:7px 0;border-radius:4px;overflow:hidden;background:#eee}.stack span{min-width:1px}.legend{display:flex;flex-wrap:wrap;gap:8px}.legend span{padding:3px 6px;background:#edf2f8;border-radius:4px}table{width:100%;border-collapse:collapse;margin-top:32px;font-size:12px}th,td{padding:6px;border-bottom:1px solid #e5eaf1;text-align:right}th:first-child,td:first-child{text-align:left}.note{color:#516172}.na{color:#8a3340}</style><main>
<h1>Hindi TV Debate Analysis</h1><p class="note">Video IDs remain fixed on every comparison axis. Threshold-dependent charts update together.</p>
<section class="controls"><label>Global threshold <select id="threshold"><option value="baseline">Baseline</option><option value="0.5">0.5 seconds</option><option value="1.0">1.0 seconds</option><option value="2.0">2.0 seconds</option><option value="5.0">5.0 seconds</option></select></label> <label> Speaker share <select id="share"><option value="original_percent">Original attributed</option><option value="clean_percent">Clean speech</option></select></label><span id="quality"></span></section>
<section class="grid" id="charts"></section><section class="card"><h2>Data quality</h2><pre id="issues"></pre></section></main><script>
const D=__DATA__; const colors=['#195c9a','#e48127','#37966f','#9b5fa9','#cc4f57','#5a8fbb','#b99c31','#5d6c84','#45a6a0'];
const valid=v=>v!==null&&v!==undefined&&Number.isFinite(+v); const fmt=v=>valid(v)?(+v).toFixed(2):'Unavailable';
function values(key,stage){return D.videos.map(v=>({id:v.video_id,value:key==='duration'?v.overview.duration_seconds/60:key==='ads'?v.overview.ads_removed:(v.stages[stage]||{})[key]}))}
function bar(title,key,stage,unit){let rows=values(key,stage), max=Math.max(1,...rows.filter(x=>valid(x.value)).map(x=>+x.value));return `<article class="card"><h2>${title}</h2><p class="note">${rows.filter(x=>valid(x.value)).length}/${rows.length} videos valid · ${unit}</p><div class="bar">${rows.map(x=>`<div class="item" title="${x.id}: ${fmt(x.value)} ${unit}"><i style="height:${valid(x.value)?100*x.value/max:0}%"></i><small>${x.id}</small></div>`).join('')}</div><table><tr><th>Video ID</th><th>${unit}</th></tr>${rows.map(x=>`<tr><td>${x.id}</td><td class="${valid(x.value)?'':'na'}">${fmt(x.value)}</td></tr>`).join('')}</table></article>`}
function shares(stage,metric){let rows=D.videos.map(v=>({id:v.video_id,s:(v.stages[stage]||{}).speaker_shares||{}}));let speakers=[...new Set(rows.flatMap(r=>Object.keys(r.s)))];return `<article class="card wide"><h2>Speaker share distribution — ${metric==='original_percent'?'original attributed':'clean speech'}</h2><p class="note">100% stacked bars; speaker IDs are local to each video.</p><div class="legend">${speakers.map((s,i)=>`<span style="border-left:10px solid ${colors[i%colors.length]}">${s}</span>`).join('')}</div>${rows.map(r=>{let sum=Object.values(r.s).reduce((a,x)=>a+(+x[metric]||0),0);return `<div><b>${r.id}</b><div class="stack">${Object.entries(r.s).map(([s,x])=>`<span title="${s}: ${fmt(x[metric])}%" style="width:${x[metric]||0}%;background:${colors[speakers.indexOf(s)%colors.length]}"></span>`).join('')}</div><small>${valid(sum)&&sum>0?`Total ${sum.toFixed(2)}%`:'Unavailable'}</small></div>`}).join('')}</article>`}
function host(stage,metric){let rows=D.videos.map(v=>({id:v.video_id,h:(v.stages[stage]||{}).host||{}}));return `<article class="card"><h2>Host speaking share</h2><p class="note">${metric==='original_percent'?'Original attributed':'Clean speech'} share; unavailable if host ID is unavailable.</p><table><tr><th>Video ID</th><th>Host</th><th>Share (%)</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td>${r.h.speaker_id||'Unavailable'}</td><td>${fmt(r.h[metric==='original_percent'?'original_share_percent':'clean_share_percent'])}</td></tr>`).join('')}</table></article>`}
function hostTurns(stage){let rows=D.videos.map(v=>({id:v.video_id,h:(v.stages[stage]||{}).host||{}}));return `<article class="card"><h2>Host speaking behaviour</h2><table><tr><th>Video ID</th><th>Longest turn (s)</th><th>Mean turn (s)</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td>${fmt(r.h.longest_seconds)}</td><td>${fmt(r.h.mean_seconds)}</td></tr>`).join('')}</table></article>`}
function render(){let s=document.querySelector('#threshold').value,m=document.querySelector('#share').value;document.querySelector('#charts').innerHTML=bar('Video duration','duration',s,'minutes')+bar('Number of speakers','speaker_count',s,'speakers')+bar('Ads removed','ads',s,'sections')+shares(s,m)+host(s,m)+hostTurns(s)+bar('Overlap events','overlap_events',s,'events')+bar('Actual overlap duration','actual_overlap_duration_seconds',s,'seconds')+bar('Clean silence duration','clean_silence_duration_seconds',s,'seconds')+bar('Total inter-turn gap duration','total_inter_turn_gap_duration_seconds',s,'seconds')+bar('Gap duration containing overlap','gaps_containing_overlap_duration_seconds',s,'seconds');}
document.querySelector('#threshold').onchange=render;document.querySelector('#share').onchange=render;document.querySelector('#quality').textContent=`${D.quality.loaded}/${D.quality.discovered} videos loaded`;document.querySelector('#issues').textContent=JSON.stringify(D.quality,null,2);render();</script></html>'''


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/v2/debate_only/videos"))
    p.add_argument("--output-dir", type=Path, default=Path("data/v2/debate_only/dashboard"))
    args = p.parse_args(); issues: list[dict] = []
    dirs = sorted(path.parent for path in args.data_dir.glob("*/final.json"))
    videos = [build_video(directory, issues) for directory in dirs]
    data = {"schema_version": "multi-video-dashboard-1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "thresholds": list(STAGES), "videos": videos,
            "metric_sources": {"duration": "comparison.stage_results.original.timeline_statistics.episode_duration_seconds", "speaker_count": "comparison.stage_results.<stage>.speaker_participation.by_speaker", "shares": "comparison.stage_results.<stage>.speaker_participation.by_speaker", "host": "comparison.stage_results.<stage>.host_analysis plus stage final.json speaker_turns", "overlap": "comparison.stage_results.<stage>.overlap_statistics", "clean_silence": "comparison.stage_results.<stage>.gap_analysis", "ads_removed": "advertisement_analysis/advertisement_exclusions.json summary.total_excluded_advertisements"},
            "quality": {"discovered": len(dirs), "loaded": sum(v["available"] for v in videos), "issues": issues}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multi_video_dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir / "multi_video_research_dashboard.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    print(f"[OK] {len(videos)} videos: {args.output_dir / 'multi_video_research_dashboard.html'}")


if __name__ == "__main__": main()
