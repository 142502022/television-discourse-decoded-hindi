from v2.ads.hindi_ads import exclusion_record, filter_transcript


def segment(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_high_confidence_twenty_second_ad_is_removed_at_configured_minimum():
    transcript = {"segments": [segment(10, 30, "buy now: price discount, call now for free delivery")]}
    record = exclusion_record("video", transcript)
    assert record["advertisement_detection"]["duration_threshold_seconds"] == 15.0
    assert len(record["excluded_advertisements"]) == 1
    assert filter_transcript(transcript, record["excluded_advertisements"])["segments"] == []


def test_short_ad_and_long_debate_speech_are_not_removed_by_duration_alone():
    short_ad = {"segments": [segment(10, 24.9, "buy now: price discount, call now for free delivery")]}
    debate = {"segments": [segment(30, 50, "सरकार की नीति पर विपक्ष का सवाल है और हम जवाब देंगे")]}
    assert not exclusion_record("video", short_ad)["excluded_advertisements"]
    assert not exclusion_record("video", debate)["excluded_advertisements"]
