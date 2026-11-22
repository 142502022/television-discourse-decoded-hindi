"""Detect long Hindi ad blocks from ASR and create timestamp-preserving debate inputs."""
import argparse
import shutil
from pathlib import Path
from v2.ads.hindi_ads import MIN_ADVERTISEMENT_DURATION_SECONDS, exclusion_record, filter_transcript, mask_audio, save_record
from v2.io import load_json, save_json
from v2.media.layout import build_layout

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--video-id",required=True); p.add_argument("--data-dir",type=Path,default=Path("data/v2/videos")); p.add_argument("--debate-data-dir",type=Path,default=Path("data/v2/debate_only/videos")); p.add_argument("--min-duration-seconds", type=float, default=MIN_ADVERTISEMENT_DURATION_SECONDS, help="Minimum high-confidence advertisement block duration to remove."); args=p.parse_args()
    source=build_layout(args.video_id,args.data_dir); debate=build_layout(args.video_id,args.debate_data_dir)
    transcript=load_json(source.asr_path); record=exclusion_record(args.video_id,transcript, args.min_duration_seconds)
    save_record(record,source.root/"advertisement_analysis"/"advertisement_exclusions.json")
    save_record(record,debate.root/"advertisement_analysis"/"advertisement_exclusions.json")
    save_json(filter_transcript(transcript,record["excluded_advertisements"]),debate.asr_path)
    mask_audio(source.audio_path,debate.audio_path,record["excluded_advertisements"])
    debate.metadata_dir.mkdir(parents=True,exist_ok=True); shutil.copy2(source.metadata_path,debate.metadata_path)
    save_record({"video_id":args.video_id,"source_audio":str(source.audio_path),"debate_audio":str(debate.audio_path),"timestamp_domain":"original_media","excluded_intervals":record["excluded_advertisements"]},debate.root/"advertisement_analysis"/"advertisement_filtered_segments.json")
    print(f"[OK] {args.video_id}: {len(record['excluded_advertisements'])} excluded ad blocks")
if __name__=="__main__": main()
