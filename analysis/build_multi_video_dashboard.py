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


def valid_speakers(final: dict | None) -> list[str]:
    """Return local speakers with at least one valid attributed turn."""
    speakers = set()
    for turn in (final or {}).get("speaker_turns", []):
        if not isinstance(turn, dict) or not isinstance(turn.get("speaker"), str):
            continue
        start, end = number(turn.get("start")), number(turn.get("end"))
        if start is not None and end is not None and end > start:
            speakers.add(turn["speaker"])
    return sorted(speakers)


def pairwise_transition_analysis(transition_data: dict | None, final: dict | None,
                                 host_id: str | None, threshold: str) -> dict:
    """Aggregate existing confirmed transition records; never infer transitions."""
    speakers = valid_speakers(final)
    matrix = {speaker: {other: 0 for other in speakers} for speaker in speakers}
    duration_seconds = number((final or {}).get("episode", {}).get("duration_s"))
    duration_minutes = duration_seconds / 60 if duration_seconds and duration_seconds > 0 else None
    source = ((transition_data or {}).get("transition_analysis", {}).get("confirmed_transitions", []))
    source = source if isinstance(source, list) else []
    records, seen, excluded = [], set(), []
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            excluded.append({"index": index, "reason": "not an object"}); continue
        left, right, timestamp = row.get("from_speaker"), row.get("to_speaker"), number(row.get("timestamp"))
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            excluded.append({"index": index, "reason": "invalid or same-speaker transition"}); continue
        if left not in matrix or right not in matrix:
            excluded.append({"index": index, "reason": "speaker absent from stage final.json"}); continue
        if timestamp is None:
            excluded.append({"index": index, "reason": "invalid timestamp"}); continue
        key = (left, right, timestamp, row.get("section_index"))
        if key in seen:
            excluded.append({"index": index, "reason": "duplicate transition record"}); continue
        seen.add(key)
        category = None
        if host_id in speakers:
            category = "HOST_TO_GUEST" if left == host_id else "GUEST_TO_HOST" if right == host_id else "GUEST_TO_GUEST"
        records.append({"from_speaker": left, "to_speaker": right, "timestamp": timestamp,
                        "section_index": row.get("section_index"), "category": category})
        matrix[left][right] += 1
    total = len(records)
    category_counts = {name: None for name in ("HOST_TO_GUEST", "GUEST_TO_HOST", "GUEST_TO_GUEST")}
    if host_id in speakers:
        category_counts = {name: sum(row["category"] == name for row in records) for name in category_counts}
    def rate(count: int | None) -> float | None:
        return round(count / duration_minutes, 3) if count is not None and duration_minutes else None
    percentages = {}
    for speaker, row in matrix.items():
        outgoing = sum(row.values())
        percentages[speaker] = {other: round(value / outgoing * 100, 3) if outgoing else None for other, value in row.items()}
    return {"threshold": threshold, "host_speaker_id": host_id, "debate_duration_minutes": round(duration_minutes, 3) if duration_minutes else None,
            "speaker_ids": speakers, "total_valid_transitions": total, "overall_transition_rate_per_minute": rate(total),
            "host_to_guest_count": category_counts["HOST_TO_GUEST"], "host_to_guest_rate_per_minute": rate(category_counts["HOST_TO_GUEST"]),
            "guest_to_host_count": category_counts["GUEST_TO_HOST"], "guest_to_host_rate_per_minute": rate(category_counts["GUEST_TO_HOST"]),
            "guest_to_guest_count": category_counts["GUEST_TO_GUEST"], "guest_to_guest_rate_per_minute": rate(category_counts["GUEST_TO_GUEST"]),
            "pairwise_transition_counts": matrix, "pairwise_transition_percentages": percentages,
            "transition_records": records,
            "data_quality": {"source": "speaker_transition_analysis.json.transition_analysis.confirmed_transitions",
                             "excluded_record_count": len(excluded), "excluded_records": excluded,
                             "host_mapping_available": host_id in speakers,
                             "category_total_matches": (sum(category_counts.values()) == total) if host_id in speakers else None,
                             "matrix_total_matches": sum(sum(row.values()) for row in matrix.values()) == total}}


def build_video(video_dir: Path, issues: list[dict]) -> dict:
    comparison = load(video_dir / "comparison" / "five_stage_comparison.json", issues, "comparison")
    if not comparison:
        return {"video_id": video_dir.name, "stages": {}, "available": False}
    result = {"video_id": video_dir.name, "available": True, "stages": {}, "overview": {}}
    raw_stages = comparison.get("stage_results", {})
    source = raw_stages.get("original", {})
    result["overview"]["duration_seconds"] = source.get("timeline_statistics", {}).get("episode_duration_seconds")
    source_speakers = source.get("speaker_participation", {}).get("by_speaker", {})
    # Speaker identity is tied to original diarization turns, not a threshold.
    result["overview"]["speaker_count"] = len(source_speakers) if isinstance(source_speakers, dict) and source_speakers else None
    ads = load(video_dir / "advertisement_analysis" / "advertisement_exclusions.json", issues, "advertisements")
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
        stage_dir = video_dir / ("baseline" if selector == "baseline" else f"threshold_{selector}")
        host_source = stage_dir / "host_identification" / "host_identification.json"
        host_data = load(host_source, issues, f"{selector}.host identification")
        host_id = host_data.get("predicted_host_speaker") if host_data else None
        host_id_matches_video = bool(host_data and host_data.get("video_id") == video_dir.name)
        if not host_id_matches_video:
            host_id = None
        final = load(stage_dir / "final.json", issues, f"{selector}.stage final")
        transitions = load(stage_dir / "speaker_transition_analysis.json", issues, f"{selector}.speaker transitions")
        gap = stage.get("gap_analysis", {})
        overlap = stage.get("overlap_statistics", {})
        host = by_speaker.get(host_id, {}) if host_id else {}
        result["stages"][selector] = {
            "available": True,
            "speaker_shares": {speaker: {"original_percent": row.get("speech_share_percent_of_attributed_total"),
                                            "clean_percent": row.get("clean_speech_share_percent"),
                                            "attributed_seconds": row.get("attributed_speaking_seconds"),
                                            "clean_seconds": row.get("clean_speech_seconds"),
                                            "excluded_seconds": row.get("speech_excluded_by_retained_overlap_seconds")}
                               for speaker, row in by_speaker.items()},
            "host": {
                "speaker_id": host_id,
                "host_id_source": str(host_source),
                "host_id_matches_video": host_id_matches_video,
                "host_found_in_analysis": bool(host_id and host_id in by_speaker),
                "attributed_speaking_duration_seconds": host.get("attributed_speaking_seconds") if host_id else None,
                "original_share_percent": host.get("speech_share_percent_of_attributed_total") if host_id else None,
                "clean_speaking_duration_seconds": host.get("clean_speech_seconds") if host_id else None,
                "clean_share_percent": host.get("clean_speech_share_percent") if host_id else None,
                **host_turn_stats(final, host_id),
            },
            "pairwise_transitions": pairwise_transition_analysis(transitions, final, host_id, selector),
            "overlap_events": overlap.get("total_overlap_events"),
            "actual_overlap_duration_seconds": overlap.get("duration", {}).get("total"),
            "clean_silence_duration_seconds": gap.get("clean_silence_duration"),
            "total_inter_turn_gap_duration_seconds": gap.get("total_inter_turn_gap_duration", {}).get("total"),
            "gaps_containing_overlap_duration_seconds": gap.get("gaps_containing_overlap_duration"),
            "clean_silence_gap_count": gap.get("clean_silence_gap_count"),
            "gaps_containing_overlap_count": gap.get("gaps_containing_overlap_count"),
        }
    return result


def prepare_dashboard_data(data_dir: Path) -> dict:
    """Read comparison artifacts only and produce the self-contained dashboard payload."""
    issues: list[dict] = []
    dirs = sorted(path.parent for path in data_dir.glob("*/final.json"))
    videos = [build_video(directory, issues) for directory in dirs]
    incomplete = []
    for video in videos:
        missing = [stage for stage in STAGES if not video["stages"].get(stage, {}).get("available")]
        if not video["available"] or missing:
            incomplete.append({"video_id": video["video_id"], "missing_threshold_stages": missing,
                               "comparison_available": video["available"]})
    invalid_json = [issue for issue in issues if "Expecting" in issue["reason"] or "JSON" in issue["reason"]]
    return {
        "schema_version": "multi-video-dashboard-1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": list(STAGES), "videos": videos,
        "metric_sources": {
            "duration": "comparison.stage_results.original.timeline_statistics.episode_duration_seconds",
            "speaker_count": "comparison.stage_results.original.speaker_participation.by_speaker (unique local diarization labels)",
            "shares": "comparison.stage_results.<stage>.speaker_participation.by_speaker",
            "host": "<stage>/host_identification/host_identification.json predicted_host_speaker, matched to that JSON's video_id; speaker shares and turns are sourced from the same stage.",
            "host_turns": "<stage>/final.json speaker_turns for the matching host speaker ID",
            "pairwise_transitions": "<stage>/speaker_transition_analysis.json transition_analysis.confirmed_transitions; only confirmed, distinct-speaker records",
            "transition_duration": "<stage>/final.json episode.duration_s (already debate-only; no advertisement subtraction)",
            "overlap": "comparison.stage_results.<stage>.overlap_statistics.duration.total (sum of stage-local candidate intervals)",
            "clean_silence": "comparison.stage_results.<stage>.gap_analysis.clean_silence_duration",
            "ads_removed": "advertisement_analysis/advertisement_exclusions.json summary.total_excluded_advertisements",
        },
        "quality": {
            "discovered": len(dirs), "loaded": sum(v["available"] for v in videos),
            "incomplete_videos": incomplete, "invalid_json": invalid_json, "issues": issues,
            "note": "Host IDs are resolved independently for each video and selected stage from its host-identification JSON.",
        },
    }


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Hindi TV Debate Research Dashboard</title><style>
body{font:14px system-ui,sans-serif;margin:0;color:#172033;background:#f6f8fb}main{max-width:1500px;margin:auto;padding:28px}.controls,.card{background:white;border:1px solid #dce3ee;border-radius:10px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}.wide{grid-column:1/-1}.bar{display:flex;align-items:end;gap:7px;height:230px;border-bottom:1px solid #9aa7b8;padding:12px 5px}.item{flex:1;min-width:28px;height:100%;display:flex;align-items:end;justify-content:center;position:relative}.item i{width:75%;background:#2463a5;border-radius:4px 4px 0 0}.item small{position:absolute;bottom:-36px;writing-mode:vertical-rl;transform:rotate(180deg);max-height:70px;overflow:hidden}.stack{display:flex;height:42px;margin:7px 0;border-radius:4px;overflow:hidden;background:#eee}.stack span{min-width:1px}.legend{display:flex;flex-wrap:wrap;gap:8px}.legend span{padding:3px 6px;background:#edf2f8;border-radius:4px}table{width:100%;border-collapse:collapse;margin-top:32px;font-size:12px}th,td{padding:6px;border-bottom:1px solid #e5eaf1;text-align:right}th:first-child,td:first-child{text-align:left}.note{color:#516172}.na{color:#8a3340}.heat-wrap{overflow:auto}.heatmap{width:auto;min-width:600px;margin-top:12px}.heatmap td{min-width:70px;text-align:center;color:#10223a}.heatmap th{white-space:nowrap}</style><main>
<h1>Hindi TV Debate Analysis</h1><p class="note">Video IDs remain fixed on every comparison axis. Threshold-dependent charts update together.</p>
<section class="controls"><label>Global threshold <select id="threshold"><option value="baseline">Baseline</option><option value="0.5">0.5 seconds</option><option value="1.0">1.0 seconds</option><option value="2.0">2.0 seconds</option><option value="5.0">5.0 seconds</option></select></label> <label> Speaker share <select id="share"><option value="original_percent">Original attributed</option><option value="clean_percent">Clean speech</option></select></label> <label>Heatmap video <select id="transition-video"></select></label> <label>Heatmap mode <select id="transition-mode"><option value="count">Raw transition count</option><option value="percent">Row-normalized percentage</option></select></label><span id="quality"></span></section>
<section class="grid" id="charts"></section><section class="card"><h2>Data quality</h2><pre id="issues"></pre></section></main><script>
const D=__DATA__; const colors=['#195c9a','#e48127','#37966f','#9b5fa9','#cc4f57','#5a8fbb','#b99c31','#5d6c84','#45a6a0'];
const valid=v=>v!==null&&v!==undefined&&Number.isFinite(+v); const fmt=v=>valid(v)?(+v).toFixed(2):'Unavailable';
function values(key,stage){return D.videos.map(v=>({id:v.video_id,value:key==='duration'?v.overview.duration_seconds/60:key==='ads'?v.overview.ads_removed:key==='speaker_count'?v.overview.speaker_count:(v.stages[stage]||{})[key]}))}
function bar(title,key,stage,unit){let rows=values(key,stage), max=Math.max(1,...rows.filter(x=>valid(x.value)).map(x=>+x.value));return `<article class="card"><h2>${title}</h2><p class="note">${rows.filter(x=>valid(x.value)).length}/${rows.length} videos valid · ${unit}</p><div class="bar">${rows.map(x=>`<div class="item" title="${x.id}: ${fmt(x.value)} ${unit}"><i style="height:${valid(x.value)?100*x.value/max:0}%"></i><small>${x.id}</small></div>`).join('')}</div><table><tr><th>Video ID</th><th>${unit}</th></tr>${rows.map(x=>`<tr><td>${x.id}</td><td class="${valid(x.value)?'':'na'}">${fmt(x.value)}</td></tr>`).join('')}</table></article>`}
function shares(stage,metric){let rows=D.videos.map(v=>({id:v.video_id,s:(v.stages[stage]||{}).speaker_shares||{}}));let speakers=[...new Set(rows.flatMap(r=>Object.keys(r.s)))];return `<article class="card wide"><h2>Speaker share distribution — ${metric==='original_percent'?'original attributed':'clean speech'}</h2><p class="note">100% stacked bars; speaker IDs are local to each video.</p><div class="legend">${speakers.map((s,i)=>`<span style="border-left:10px solid ${colors[i%colors.length]}">${s}</span>`).join('')}</div>${rows.map(r=>{let sum=Object.values(r.s).reduce((a,x)=>a+(+x[metric]||0),0);return `<div><b>${r.id}</b><div class="stack">${Object.entries(r.s).map(([s,x])=>`<span title="${s}: ${fmt(x[metric])}%" style="width:${x[metric]||0}%;background:${colors[speakers.indexOf(s)%colors.length]}"></span>`).join('')}</div><small>${valid(sum)&&sum>0?`Total ${sum.toFixed(2)}%`:'Unavailable'}</small></div>`}).join('')}</article>`}
function host(stage,metric){let rows=D.videos.map(v=>({id:v.video_id,h:(v.stages[stage]||{}).host||{}})),key=metric==='original_percent'?'original_share_percent':'clean_share_percent';let max=Math.max(1,...rows.filter(r=>valid(r.h[key])).map(r=>+r.h[key]));return `<article class="card"><h2>Host Speaking Share by Video</h2><p class="note">${metric==='original_percent'?'Original attributed':'Clean-speech'} share; ${rows.filter(r=>valid(r.h[key])).length}/${rows.length} videos valid.</p><div class="bar">${rows.map(r=>`<div class="item" title="${r.id}: ${fmt(r.h[key])}%"><i style="height:${valid(r.h[key])?100*r.h[key]/max:0}%"></i><small>${r.id}</small></div>`).join('')}</div><table><tr><th>Video ID</th><th>Host</th><th>Speech (s)</th><th>Share (%)</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td>${r.h.speaker_id||'Unavailable'}</td><td class="${valid(r.h[metric==='original_percent'?'attributed_speaking_duration_seconds':'clean_speaking_duration_seconds'])?'':'na'}">${fmt(r.h[metric==='original_percent'?'attributed_speaking_duration_seconds':'clean_speaking_duration_seconds'])}</td><td class="${valid(r.h[key])?'':'na'}">${fmt(r.h[key])}</td></tr>`).join('')}</table></article>`}
function hostTurns(stage){let rows=D.videos.map(v=>({id:v.video_id,h:(v.stages[stage]||{}).host||{}})),max=Math.max(1,...rows.flatMap(r=>[r.h.longest_seconds,r.h.mean_seconds]).filter(valid).map(Number));return `<article class="card"><h2>Host Speaking Behaviour by Video</h2><p class="note">Longest and average continuous host turn; ${rows.filter(r=>valid(r.h.longest_seconds)&&valid(r.h.mean_seconds)).length}/${rows.length} videos valid.</p><div class="bar">${rows.map(r=>`<div class="item" title="${r.id}: longest ${fmt(r.h.longest_seconds)} s; average ${fmt(r.h.mean_seconds)} s"><i style="height:${valid(r.h.longest_seconds)?100*r.h.longest_seconds/max:0}%;width:34%;background:#2463a5"></i><i style="height:${valid(r.h.mean_seconds)?100*r.h.mean_seconds/max:0}%;width:34%;background:#e48127"></i><small>${r.id}</small></div>`).join('')}</div><p class="note">Blue: longest turn; orange: average turn.</p><table><tr><th>Video ID</th><th>Longest turn (s)</th><th>Average turn (s)</th><th>Turn count</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td class="${valid(r.h.longest_seconds)?'':'na'}">${fmt(r.h.longest_seconds)}</td><td class="${valid(r.h.mean_seconds)?'':'na'}">${fmt(r.h.mean_seconds)}</td><td>${r.h.turn_count||'Unavailable'}</td></tr>`).join('')}</table></article>`}
function hostValidation(stage){let rows=D.videos.map(v=>({id:v.video_id,h:(v.stages[stage]||{}).host||{}}));return `<article class="card wide"><h2>Host mapping validation</h2><table><tr><th>Video ID</th><th>Host ID</th><th>Host ID source</th><th>Host found in analysis</th><th>Speaking share available</th><th>Behaviour available</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td>${r.h.speaker_id||'Unavailable'}</td><td>${r.h.host_id_source||'Unavailable'}</td><td>${r.h.host_found_in_analysis?'Yes':'No'}</td><td>${valid(r.h.original_share_percent)&&valid(r.h.clean_share_percent)?'Yes':'No'}</td><td>${valid(r.h.longest_seconds)&&valid(r.h.mean_seconds)?'Yes':'No'}</td></tr>`).join('')}</table></article>`}
function transitionBar(title,key,stage){let rows=D.videos.map(v=>({id:v.video_id,p:(v.stages[stage]||{}).pairwise_transitions||{}})),max=Math.max(1,...rows.filter(r=>valid(r.p[key])).map(r=>+r.p[key]));return `<article class="card"><h2>${title}</h2><p class="note">${rows.filter(r=>valid(r.p[key])).length}/${rows.length} videos valid · transitions per minute</p><div class="bar">${rows.map(r=>`<div class="item" title="${r.id}: ${fmt(r.p[key])} transitions/min; raw ${fmt(key==='overall_transition_rate_per_minute'?r.p.total_valid_transitions:r.p[key.replace('_rate_per_minute','_count')])}"><i style="height:${valid(r.p[key])?100*r.p[key]/max:0}%"></i><small>${r.id}</small></div>`).join('')}</div><table><tr><th>Video ID</th><th>Rate (transitions/min)</th><th>Raw count</th></tr>${rows.map(r=>`<tr><td>${r.id}</td><td class="${valid(r.p[key])?'':'na'}">${fmt(r.p[key])}</td><td>${fmt(key==='overall_transition_rate_per_minute'?r.p.total_valid_transitions:r.p[key.replace('_rate_per_minute','_count')])}</td></tr>`).join('')}</table></article>`}
function transitionCategoryComparison(stage){let rows=D.videos.map(v=>({id:v.video_id,p:(v.stages[stage]||{}).pairwise_transitions||{}})),keys=['host_to_guest_rate_per_minute','guest_to_host_rate_per_minute','guest_to_guest_rate_per_minute'],max=Math.max(1,...rows.flatMap(r=>keys.map(k=>r.p[k])).filter(valid).map(Number)),colors2=['#2463a5','#e48127','#37966f'];return `<article class="card wide"><h2>Transition category comparison</h2><p class="note">Rates per minute. Blue: Host → Guest; orange: Guest → Host; green: Guest → Guest.</p><div class="bar">${rows.map(r=>`<div class="item" title="${r.id}: H→G ${fmt(r.p.host_to_guest_rate_per_minute)}, G→H ${fmt(r.p.guest_to_host_rate_per_minute)}, G→G ${fmt(r.p.guest_to_guest_rate_per_minute)} transitions/min"><i style="height:${valid(r.p.host_to_guest_rate_per_minute)?100*r.p.host_to_guest_rate_per_minute/max:0}%;width:25%;background:${colors2[0]}"></i><i style="height:${valid(r.p.guest_to_host_rate_per_minute)?100*r.p.guest_to_host_rate_per_minute/max:0}%;width:25%;background:${colors2[1]}"></i><i style="height:${valid(r.p.guest_to_guest_rate_per_minute)?100*r.p.guest_to_guest_rate_per_minute/max:0}%;width:25%;background:${colors2[2]}"></i><small>${r.id}</small></div>`).join('')}</div></article>`}
function transitionHeatmap(stage){let id=document.querySelector('#transition-video').value, mode=document.querySelector('#transition-mode').value,video=D.videos.find(v=>v.video_id===id),p=video&&(video.stages[stage]||{}).pairwise_transitions;if(!p)return `<article class="card wide"><h2>Individual-video pairwise transition heatmap</h2><p class="na">Unavailable</p></article>`;let speakers=p.speaker_ids||[],host=p.host_speaker_id,max=Math.max(1,...speakers.flatMap(a=>speakers.map(b=>mode==='count'?p.pairwise_transition_counts[a][b]:p.pairwise_transition_percentages[a][b])).filter(valid).map(Number)),cell=(a,b)=>{let v=mode==='count'?p.pairwise_transition_counts[a][b]:p.pairwise_transition_percentages[a][b],label=mode==='count'?fmt(v):valid(v)?fmt(v)+'%':'No outgoing transitions',opacity=valid(v)?.12+.78*(+v/max):0;return `<td title="${a} → ${b}: ${label}" style="background:rgba(25,92,154,${opacity})">${a===b?'—':label}</td>`};let ok=p.data_quality.category_total_matches;return `<article class="card wide"><h2>Individual-video pairwise transition heatmap</h2><p class="note">${id} · ${stage==='baseline'?'Baseline':stage+' seconds'} · Host: ${host||'Unavailable'} · ${mode==='count'?'raw counts':'row-normalized percentages'}</p><table><tr><th>Speakers</th><th>Duration (min)</th><th>Transitions</th><th>Overall/min</th><th>H→G /min</th><th>G→H /min</th><th>G→G /min</th><th>Category check</th></tr><tr><td>${speakers.length}</td><td>${fmt(p.debate_duration_minutes)}</td><td>${fmt(p.total_valid_transitions)}</td><td>${fmt(p.overall_transition_rate_per_minute)}</td><td>${fmt(p.host_to_guest_rate_per_minute)}</td><td>${fmt(p.guest_to_host_rate_per_minute)}</td><td>${fmt(p.guest_to_guest_rate_per_minute)}</td><td>${ok===true?'Pass':ok===false?'Mismatch':'Unavailable'}</td></tr></table><div class="heat-wrap"><table class="heatmap"><tr><th>From / To</th>${speakers.map(s=>`<th>${s}${s===host?' (Host)':''}</th>`).join('')}</tr>${speakers.map(a=>`<tr><th>${a}${a===host?' (Host)':''}</th>${speakers.map(b=>cell(a,b)).join('')}</tr>`).join('')}</table></div></article>`}
function render(){let s=document.querySelector('#threshold').value,m=document.querySelector('#share').value;document.querySelector('#charts').innerHTML=bar('Video duration','duration',s,'minutes')+bar('Number of speakers','speaker_count',s,'speakers')+bar('Ads removed','ads',s,'sections')+shares(s,m)+host(s,m)+hostTurns(s)+hostValidation(s)+bar('Overlap events','overlap_events',s,'events')+bar('Actual overlap duration','actual_overlap_duration_seconds',s,'seconds')+bar('Clean silence duration','clean_silence_duration_seconds',s,'seconds')+bar('Total inter-turn gap duration','total_inter_turn_gap_duration_seconds',s,'seconds')+bar('Gap duration containing overlap','gaps_containing_overlap_duration_seconds',s,'seconds')+transitionBar('Overall speaker transitions per minute','overall_transition_rate_per_minute',s)+transitionBar('Host → Guest transitions per minute','host_to_guest_rate_per_minute',s)+transitionBar('Guest → Host transitions per minute','guest_to_host_rate_per_minute',s)+transitionBar('Guest → Guest transitions per minute','guest_to_guest_rate_per_minute',s)+transitionCategoryComparison(s)+transitionHeatmap(s)}
document.querySelector('#transition-video').innerHTML=D.videos.map(v=>`<option value="${v.video_id}">${v.video_id}</option>`).join('');document.querySelector('#threshold').onchange=render;document.querySelector('#share').onchange=render;document.querySelector('#transition-video').onchange=render;document.querySelector('#transition-mode').onchange=render;document.querySelector('#quality').textContent=`${D.quality.loaded}/${D.quality.discovered} videos loaded`;document.querySelector('#issues').textContent=JSON.stringify(D.quality,null,2);render();</script></html>'''


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/v2/debate_only/videos"))
    p.add_argument("--output-dir", type=Path, default=Path("data/v2/debate_only/dashboard"))
    args = p.parse_args()
    data = prepare_dashboard_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multi_video_dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir / "multi_video_research_dashboard.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    print(f"[OK] {len(data['videos'])} videos: {args.output_dir / 'multi_video_research_dashboard.html'}")


if __name__ == "__main__": main()
