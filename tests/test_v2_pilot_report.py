from pathlib import Path

from pilot_report import ReportPaths, load_sources, render


def test_render_with_no_artifacts_marks_missing_but_never_raises():
    markdown = render({}, ReportPaths())
    assert "# Pilot Episode Report" in markdown
    assert "MISSING" in markdown
    assert "## Reproduction" in markdown
    assert "make drift" in markdown


def test_render_includes_host_prediction_when_present():
    sources = {"host": {"prediction": {"speaker_label": "SPEAKER_24", "inferred_role": "host_moderator", "confidence_band": "high"},
                        "reference_comparison": {"reference_label": "SPEAKER_24", "match": True, "reference_rank": 1},
                        "candidates": [{"rank": 1, "speaker_label": "SPEAKER_24", "heuristic_score": 1.0, "inferred_role": "host_moderator"}]}}
    markdown = render(sources, ReportPaths())
    assert "SPEAKER_24" in markdown
    assert "host_moderator" in markdown
    assert "| 1 | `SPEAKER_24` |" in markdown


def test_render_flags_stale_empty_segmentation_sample():
    sources = {"topic_segmentation": {"aggregate": {"processing": {"analyzed": 1, "skipped": 0, "failed": 0},
                                                    "topic_segments": {"broad_total": 0, "subtopic_total": 0},
                                                    "shift_events": {"total": 0}}}}
    markdown = render(sources, ReportPaths())
    assert "Stale/empty sample detected" in markdown


def test_render_drift_table_lists_lift():
    sources = {"drift": {"aggregate": {"interruption_events": 313,
                                       "comparisons": {"45": {"subtopic": {"interruption_rate": 0.22, "non_interruption_rate": 0.25, "lift": -0.03}}}},
                        "episodes": [{"timeline": {"assessed_events": 302, "insufficient_context_events": 11}}]}}
    markdown = render(sources, ReportPaths())
    assert "| 45 s | subtopic | 0.22 | 0.25 | -0.03 |" in markdown


def test_render_discourse_cross_check_line():
    sources = {"discourse": {"aggregate": {"processing": {"successfully_analyzed": 1, "skipped": 0}}},
               "discourse_review": {"aggregate": {"cross_check": {"matched": 18, "metrics_compared": 18, "mismatched": 0}},
                                    "episodes": [{"structural_summary": {"out_of_order": 0, "zero_duration": 0, "duplicates": 0},
                                                  "validation": {"overlap_candidates": {"nested": 10}}}]}}
    markdown = render(sources, ReportPaths())
    assert "18/18 metrics matched (0 mismatched)" in markdown
    assert "nested overlaps=10" in markdown


def test_load_sources_handles_missing_and_present(tmp_path):
    present = tmp_path / "present.json"
    present.write_text('{"schema_version": "x"}', encoding="utf-8")
    paths = ReportPaths(audit=present, host=tmp_path / "absent.json")
    sources = load_sources(paths)
    assert sources["audit"] == {"schema_version": "x"}
    assert sources["host"] is None
