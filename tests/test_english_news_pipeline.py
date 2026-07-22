from src.speech.utils import diarization_intersections, merge_overlap_candidates


def test_diarization_intersections_keeps_overlapping_speakers():
    intervals = diarization_intersections(
        [
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 6.0, "end": 7.0, "speaker": "SPEAKER_00"},
        ]
    )

    assert intervals == [
        {
            "start": 3.0,
            "end": 4.0,
            "source": "diarization_intersection",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
        }
    ]


def test_merge_overlap_candidates_marks_agreement_high_confidence():
    candidates = merge_overlap_candidates(
        [{"start": 10.0, "end": 12.0}],
        [
            {
                "start": 10.2,
                "end": 11.9,
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
            }
        ],
    )

    assert candidates[0]["confidence"] == "high"
    assert candidates[0]["sources"] == [
        "pyannote_osd",
        "diarization_intersection",
    ]


def test_merge_overlap_candidates_keeps_osd_only_interval():
    candidates = merge_overlap_candidates(
        [{"start": 10.0, "end": 12.0}],
        [],
    )

    assert candidates[0]["confidence"] == "medium"
    assert candidates[0]["reason"] == "pyannote_osd_only"
