#!/usr/bin/env python3
"""Stage 6: complete speaker interaction analysis for one transcript variant.

Uses existing structural transition events and aggregate OSD overlap/interruption
statistics. It deliberately does not infer interruptions from overlap or transitions.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 'host-interaction-1.0'


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return obj


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def assign_subtopic(ts: float, subtopics: list[dict]) -> Optional[str]:
    """Assign by event timestamp; first containing interval wins at shared edges."""
    for st in subtopics:
        a, b = num(st.get('start')), num(st.get('end'))
        if a is not None and b is not None and a <= ts < b:
            return st.get('subtopic_id')
    # Include an event exactly at the final endpoint in the last subtopic.
    if subtopics:
        last = subtopics[-1]
        if num(last.get('end')) == ts:
            return last.get('subtopic_id')
    return None


def build_analysis(video_id: str, variant: str, transcript: dict, numeric: dict,
                   host_data: dict, subtopic_data: dict) -> dict:
    if variant != 'threshold_5.0':
        raise ValueError('Stage 6 is restricted to threshold_5.0')
    host = host_data.get('predicted_host_speaker')
    if not host:
        raise ValueError('predicted_host_speaker missing from host identification JSON')

    turns = []
    for i, t in enumerate(transcript.get('speaker_turns', [])):
        if not isinstance(t, dict):
            continue
        s, e, sp = num(t.get('start')), num(t.get('end')), t.get('speaker')
        if s is not None and e is not None and e > s and sp:
            turns.append({'turn_index': i, 'speaker_id': str(sp), 'start': s, 'end': e})
    turns.sort(key=lambda x: (x['start'], x['end']))
    speakers = sorted({t['speaker_id'] for t in turns})

    duration = num(transcript.get('episode', {}).get('duration_s'))
    if duration is None or duration <= 0:
        duration = num(numeric.get('numeric_summary', {}).get('episode_overview.duration_seconds'))
    if duration is None or duration <= 0:
        duration = max((t['end'] for t in turns), default=0.0)

    # Prefer the project's existing, documented structural transition events.
    final_result_path = numeric.get('_final_result')
    final_result = numeric.get('_final_result_obj', {})
    transition_section = final_result.get('speaker_transitions', {})
    source_events = transition_section.get('transition_events', [])
    events = []
    for i, ev in enumerate(source_events):
        frm, to = ev.get('from_speaker'), ev.get('to_speaker')
        ts = num(ev.get('timestamp'))
        if not frm or not to or ts is None:
            continue
        events.append({
            'event_id': f'TR{i+1:05d}', 'from_speaker': frm, 'to_speaker': to,
            'timestamp': ts, 'gap_seconds': num(ev.get('gap_seconds')),
            'is_overlapping_transition': bool(ev.get('is_overlapping_transition', False)),
            'previous_turn_duration_seconds': num(ev.get('previous_turn_duration_seconds')),
            'subtopic_id': None,
            'source': 'final_result.json:speaker_transitions.transition_events',
            'classification': 'structural_transition_not_interruption'
        })
    # Fallback only if rich transition event records are unavailable.
    if not events:
        for i in range(1, len(turns)):
            prev, cur = turns[i-1], turns[i]
            if prev['speaker_id'] == cur['speaker_id']:
                continue
            events.append({'event_id': f'TR{i:05d}', 'from_speaker': prev['speaker_id'],
                'to_speaker': cur['speaker_id'], 'timestamp': cur['start'],
                'gap_seconds': cur['start']-prev['end'],
                'is_overlapping_transition': cur['start'] < prev['end'],
                'previous_turn_duration_seconds': prev['end']-prev['start'],
                'subtopic_id': None, 'source': 'derived_from_merge_final.speaker_turns',
                'classification': 'structural_transition_not_interruption'})

    subtopics = subtopic_data.get('subtopics', [])
    for ev in events:
        ev['subtopic_id'] = assign_subtopic(ev['timestamp'], subtopics)

    # Directed transition matrix includes zeros for all observed speakers.
    matrix = {a: {b: 0 for b in speakers} for a in speakers}
    pair_counts = Counter()
    for ev in events:
        a, b = ev['from_speaker'], ev['to_speaker']
        if a in matrix and b in matrix:
            matrix[a][b] += 1
            pair_counts[(a,b)] += 1

    # OSD aggregate overlap and interruption records have no timestamps in this
    # source format; keep them as aggregate evidence, never assign them to subtopics.
    osd_stats = transcript.get('osd_stats', {})
    overlap_pairs = []
    for item in osd_stats.get('by_pair', []) or []:
        pair = item.get('pair', [])
        if len(pair) == 2:
            overlap_pairs.append({'speaker_a': pair[0], 'speaker_b': pair[1],
                'overlap_count': int(item.get('count', 0)),
                'overlap_duration_seconds': num(item.get('seconds'), 0.0),
                'host_involved': host in pair,
                'source': 'merge_final.json:osd_stats.by_pair'})
    interruption_pairs = []
    for item in osd_stats.get('interruptions_by_pair', []) or []:
        a, b = item.get('interrupter'), item.get('victim')
        if a and b:
            interruption_pairs.append({'interrupter': a, 'interrupted_speaker': b,
                'count': int(item.get('count', 0)),
                'overlap_duration_seconds': num(item.get('seconds'), 0.0),
                'host_involved': host in (a,b),
                'source': 'merge_final.json:osd_stats.interruptions_by_pair'})

    # Responses are only temporal re-entry observations, not semantic answers.
    response_window = 3.0
    response_rows = []
    for ev in events:
        target = ev['from_speaker']
        next_turn = next((t for t in turns if t['start'] > ev['timestamp'] and t['speaker_id'] == target), None)
        delta = (next_turn['start'] - ev['timestamp']) if next_turn else None
        response_rows.append({'transition_event_id': ev['event_id'],
            'speaker_before_transition': target, 'speaker_taking_floor': ev['to_speaker'],
            'next_turn_by_previous_speaker_seconds_after_transition': delta,
            'previous_speaker_returns_within_3_seconds': bool(delta is not None and 0 <= delta <= response_window),
            'interpretation': 'temporal_reentry_only_not_semantically_verified_response'})

    # Subtopic transition stats, one assignment per transition timestamp.
    subtopic_rows = []
    for st in subtopics:
        sid = st.get('subtopic_id')
        subset = [e for e in events if e.get('subtopic_id') == sid]
        pc = Counter((e['from_speaker'], e['to_speaker']) for e in subset)
        subtopic_rows.append({'subtopic_id': sid, 'title': st.get('title'),
            'start': st.get('start'), 'end': st.get('end'),
            'transition_count': len(subset),
            'host_involved_transition_count': sum(host in (e['from_speaker'],e['to_speaker']) for e in subset),
            'directional_transition_counts': [{'from_speaker':a,'to_speaker':b,'count':n}
                for (a,b),n in sorted(pc.items())]})

    host_events = [e for e in events if host in (e['from_speaker'], e['to_speaker'])]
    host_out = sum(e['from_speaker'] == host for e in events)
    host_in = sum(e['to_speaker'] == host for e in events)
    episode_summary = {
        'duration_seconds': duration, 'duration_minutes': duration/60 if duration else None,
        'speaker_count': len(speakers), 'speaker_ids': speakers,
        'structural_transition_count': len(events),
        'structural_transitions_per_minute': round(len(events)/(duration/60), 4) if duration else None,
        'overlap_pair_count': sum(x['overlap_count'] for x in overlap_pairs),
        'overlap_duration_seconds_sum_by_pair': round(sum(x['overlap_duration_seconds'] for x in overlap_pairs), 6),
        'osd_interruption_count_sum_by_directional_pair': sum(x['count'] for x in interruption_pairs),
        'note': 'Pairwise overlap seconds may be summed across pairs; this is not unique wall-clock overlap duration.'}

    return {
        'schema_version': SCHEMA_VERSION,
        'metadata': {'video_id': video_id, 'variant': variant, 'host_speaker_id': host,
            'host_identification_source': f'data/v2/debate_only/videos/{video_id}/{variant}/host_identification/host_identification.json',
            'host_identification_method': host_data.get('method'),
            'host_confidence_band': host_data.get('confidence_band'),
            'host_score_warning': host_data.get('warning')},
        'methodology': {
            'transition_definition': 'A structural change from one diarized speaker turn to the next; not proof of interruption or floor allocation.',
            'overlap_definition': 'OSD aggregate overlap pair statistics are preserved separately from speaker transitions.',
            'interruption_definition': 'Only existing merge_final.json osd_stats.interruptions_by_pair labels are counted as interruptions; no new interruption labels are inferred.',
            'response_definition': 'A subsequent turn by the previous speaker within 3 seconds is temporal re-entry only, not a semantically verified response.',
            'subtopic_assignment': 'Each timestamped transition is assigned once to the subtopic containing its timestamp (start inclusive, end exclusive). Aggregate overlap/interruption records have no event timestamps and are not allocated to subtopics.',
            'floor_allocation_policy': 'Transitions involving the host are not automatically interpreted as deliberate floor allocation.'},
        'episode_summary': episode_summary,
        'speaker_interactions': {'directed_transition_matrix': matrix,
            'pairwise_directional_counts': [{'from_speaker':a,'to_speaker':b,'count':n} for (a,b),n in sorted(pair_counts.items())]},
        'transition_events': events,
        'interruption_analysis': {'status': 'aggregate_source_labels_available' if interruption_pairs else 'no_interruption_records',
            'directional_pairs': interruption_pairs,
            'total_directional_interruption_count': sum(x['count'] for x in interruption_pairs),
            'limitation': 'Source provides aggregate pair counts/durations without timestamps; event-level and subtopic-level interruption analysis is unavailable.'},
        'overlap_analysis': {'status': 'aggregate_pair_statistics_available' if overlap_pairs else 'no_overlap_pair_records',
            'directional_or_undirected_pairs': overlap_pairs,
            'total_pair_event_count': sum(x['overlap_count'] for x in overlap_pairs),
            'pair_duration_sum_seconds': round(sum(x['overlap_duration_seconds'] for x in overlap_pairs), 6),
            'limitation': 'Pair-level aggregate only; no event timestamps in merge_final.json after this variant preprocessing.'},
        'host_interaction': {'host_speaker_id': host, 'transition_count_involving_host': len(host_events),
            'host_as_previous_speaker_count': host_out, 'host_as_next_speaker_count': host_in,
            'host_related_transitions': host_events,
            'host_involved_osd_interruption_count': sum(x['count'] for x in interruption_pairs if x['host_involved']),
            'host_involved_overlap_pair_count': sum(x['overlap_count'] for x in overlap_pairs if x['host_involved']),
            'floor_allocation_events': [],
            'floor_allocation_status': 'not_inferred_from_transitions; no timestamped verified floor-allocation labels supplied'},
        'response_patterns': {'window_seconds': response_window, 'events': response_rows,
            'within_window_count': sum(x['previous_speaker_returns_within_3_seconds'] for x in response_rows),
            'limitation': 'Temporal re-entry only; does not establish that the speaker answered or responded semantically.'},
        'subtopic_interactions': subtopic_rows,
        'interaction_network': {'nodes': speakers,
            'directed_edges': [{'source':a,'target':b,'transition_count':n,
                'confirmed_interruption_count':sum(x['count'] for x in interruption_pairs if x['interrupter']==a and x['interrupted_speaker']==b),
                'pair_overlap_count':sum(x['overlap_count'] for x in overlap_pairs if set((x['speaker_a'],x['speaker_b']))==set((a,b))),
                'pair_overlap_duration_seconds':sum(x['overlap_duration_seconds'] for x in overlap_pairs if set((x['speaker_a'],x['speaker_b']))==set((a,b)))}
                for (a,b),n in sorted(pair_counts.items())]},
        'data_quality': {'transcript_speaker_turn_count': len(turns),
            'transition_events_source_count': len(events), 'subtopic_count': len(subtopics),
            'unassigned_transition_count': sum(e['subtopic_id'] is None for e in events),
            'host_id_present_in_transcript_speakers': host in speakers,
            'source_interruption_records_present': bool(interruption_pairs),
            'source_overlap_pair_records_present': bool(overlap_pairs),
            'warnings': (["Host ID is not present in transcript speaker turns."] if host not in speakers else []) +
                (["No timestamped overlap events were available; only aggregate pair statistics were used."] if overlap_pairs else []) +
                (["No interruption pair records were available in merge_final.json."] if not interruption_pairs else [])}
    }


def process_one(video_id: str, args) -> Path:
    vroot = args.videos_root / video_id / args.variant
    transcript_path = vroot / 'merge_final.json'
    numeric_path = vroot / 'final_result_numeric.json'
    host_path = vroot / 'host_identification' / 'host_identification.json'
    subtopic_path = args.subtopics_root / args.variant / video_id / 'subtopic_analysis.json'
    for p in (transcript_path, numeric_path, host_path, subtopic_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing required input: {p}')
    transcript = load_json(transcript_path)
    numeric = load_json(numeric_path)
    host = load_json(host_path)
    subtopics = load_json(subtopic_path)
    if transcript.get('episode', {}).get('video_id', video_id) != video_id:
        raise ValueError(f'Video ID mismatch in {transcript_path}')
    if host.get('video_id', video_id) != video_id or subtopics.get('video_id', video_id) != video_id:
        raise ValueError('Video ID mismatch in host or subtopic input')
    if numeric.get('video_id', video_id) != video_id:
        raise ValueError(f'Video ID mismatch in {numeric_path}')
    # The aggregate script's final_result.json is the source of detailed transition events.
    final_result_path = vroot / 'final_result.json'
    if final_result_path.exists():
        numeric['_final_result_obj'] = load_json(final_result_path)
    result = build_analysis(video_id, args.variant, transcript, numeric, host, subtopics)
    out = args.output_root / args.variant / video_id / 'host_interaction.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return out


def main():
    p = argparse.ArgumentParser(description='Stage 6 complete debate interaction analysis')
    p.add_argument('--video-id')
    p.add_argument('--all', action='store_true')
    p.add_argument('--variant', default='threshold_5.0')
    p.add_argument('--videos-root', type=Path, default=Path('data/v2/debate_only/videos'))
    p.add_argument('--subtopics-root', type=Path, default=Path('data/v2/debate_only/subtopic_analysis'))
    p.add_argument('--output-root', type=Path, default=Path('data/v2/debate_only/host_interaction'))
    args = p.parse_args()
    if args.variant != 'threshold_5.0':
        p.error('Only --variant threshold_5.0 is supported')
    if not args.all and not args.video_id:
        p.error('Provide --video-id VIDEO_ID or --all')
    ids = [args.video_id] if args.video_id else sorted(x.name for x in args.videos_root.iterdir() if x.is_dir() and (x/args.variant).is_dir())
    ok = skipped = failed = 0
    for vid in ids:
        try:
            out = process_one(vid, args); ok += 1
            print(f'[OK] {vid} -> {out}')
        except FileNotFoundError as e:
            skipped += 1; print(f'[SKIP] {vid}: {e}', file=sys.stderr)
        except Exception as e:
            failed += 1; print(f'[FAIL] {vid}: {e}', file=sys.stderr)
    print(f'\nSummary: processed={ok}, skipped={skipped}, failed={failed}')
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
