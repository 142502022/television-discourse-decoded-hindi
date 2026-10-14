# Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale

This repository contains the official code and data for [our paper](https://dl.acm.org/doi/10.1145/3637528.3671532) **"Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale"**, accepted at **KDD'2024**.

Our work introduces an automated toolkit that leverages state-of-the-art computer vision and speech-to-text techniques to transcribe, diarize, and analyze thousands of YouTube videos from televised debates, offering profound insights into biases, incivility, and the overall quality of public discourse.

## Repository Structure and Contents

### Code
* **`config_constants.py`**: Contains configurable parameters, including API keys (such as Hugging Face tokens, Perspective API keys), and paths for storing intermediate data.
* **`diarization_vad_osd_related/run_pipeline_osd_vad.py`**: Processes a video by removing segments where no voice activity is detected using Voice Activity Detection (VAD).
* **`perspective_related/run_pipeline_perspective.py`**: Analyzes the foul speech content for each utterance, assessing it across various dimensions (e.g., identity attack, profanity) based on the spoken content.
* **`transcription_related/run_pipeline_transcription.py`**: Transcribes each utterance identified during the diarization process, providing a text representation of the spoken content.
* **`tv_debs_utils/debate_utils.py`**: Contains a set of utility functions for processing, downloading, and truncating videos.


### Dataset contents
This dataset `data/video_details.json` contains metadata and labels for YouTube videos used in our work. Each entry in the dataset corresponds to a video and includes various fields detailing the video's attributes, statistics, and detected hashtags.
- **`video_idx`**: A unique index assigned to each video in the dataset.
- **`yt_vid_id`**: The unique identifier for the video on YouTube. This is the `videoId` that appears in YouTube URLs.
- **`yt_vid_url`**: The full URL to the video on YouTube.
- **`major_label`**: The primary category or theme associated with the video. This provides a high-level categorization of the video's content.
- **`minor_labels`**: A list of secondary labels or subcategories that further describe the video's content. These labels offer more granular categorization.
- **`yt_stats`**: A dictionary containing statistics related to the video on YouTube.
- **`publish_time`**: The timestamp indicating when the video was published on YouTube.
- **`vid_title`**: The title of the video as it appears on YouTube.
- **`total_duration`**: The total duration of the video in seconds.
- **`total_duration_str`**: The total duration of the video in ISO 8601 duration format.
- **`hashtags_detected`**: A list of hashtags detected in the video's description or title or via OCR on the video frames.

### Results and Intermediate files related
* **`data/scratch_folder`**
    * **`./part_0`**: Downloads the video from YouTube using the YT ID.
    * **`./part_1`**: Processes the video from `part_0`; contains the audio versions of the debates after applying Voice Activity Detection (VAD), removing segments where no voice was detected.
    * **`./part_2`**: Processes the audio from `part_1` by removing segments where more than one speaker is detected, using Overlapped Speech Detection (OSD).

* **`data/results`**
    * **`./osd_data`**: Stores timestamps for segments of the video where multiple speakers were detected, indicating overlapping speech.
    * **`./vad_data`**: Stores timestamps for segments of the video where any voice activity was detected.
    * **`./diarization_data`**: Contains timestamps for segments of the video where different speakers were detected. Includes speaker IDs, maintaining consistent identification for each speaker throughout the video, numbered from 0 to N-1, where N is the total number of speakers.
    * **`./transcription_data`**: Provides detailed information about each utterance, including the content of the speech, timestamps of the utterance, and the speaker ID associated with it.
    * **`./perspective_data`**: Contains information on any foul language or offensive content found in the transcript, with details linked to specific utterances.

The data used in this work can be found [in this zip file](https://drive.google.com/file/d/1rrhVjf7xi8BgmMhX_TBu2Wq2LX4cLT9X/view?usp=sharing). Please feel free to contact us if you have trouble accessing it.

## Getting Started

### Setting up the Project Repository

To get started with the project, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/anmolagarwal999/television-discourse-decoded
cd television-discourse-decoded
```

2. Create and activate the conda environment:
```bash
conda create --name tv_debs_env python=3.9
conda activate tv_debs_env
```
3. Install the required dependencies
```bash
pip install -r requirements.txt
```

### Usage
```bash
# To run the OSD+VAD pipeline
television-discourse-decoded> python -m src.diarization_vad_osd_related.run_pipeline_osd_vad <Youtube ID of video to process>

# To transcribe the video
television-discourse-decoded> python -m src.transcription_related.run_pipeline_transcription <Youtube ID of video to process>

# To measure foul speech in the transcribed content
television-discourse-decoded> python -m src.perspective_related.run_pipeline_perspective

```

### MTP Data Engineering Layer

This fork adds a lightweight data engineering layer around the existing ML pipeline without changing the VAD, diarization, transcription, toxicity, or gender/age model logic.

The layer lives in `src/data_engineering/` and provides:

* **PostgreSQL schema** via SQLAlchemy ORM:
  * `episodes`
  * `segments`
  * `transcripts`
  * `toxicity_scores`
  * `gender_age_labels`
* **Prefect orchestration** in `src/data_engineering/flow.py`:
  ingest episode -> VAD/diarization -> transcription -> toxicity scoring -> optional gender/age detection -> validate outputs -> store to Postgres
* **Validation contracts** in `src/data_engineering/validation.py` using Pydantic:
  non-empty transcripts, toxicity scores in `[0, 1]`, expected gender labels, plausible age estimates, and a no-orphaned-segments sanity check.
* **Unit tests** in `tests/test_data_engineering.py` for transform and validation behavior.

Set Postgres credentials with `DATABASE_URL`, for example:

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/mtp"
```

To create the database tables:

```bash
python -c "from src.data_engineering.database import create_tables; create_tables()"
```

To run the orchestration flow:

```bash
python -c "from src.data_engineering.flow import mtp_data_engineering_flow; mtp_data_engineering_flow('<Youtube ID of video to process>')"
```

The gender/age stage is optional in the flow because ad interference is currently a known blocker for full end-to-end runs.

### English News Overlap Annotation Pipeline

This branch also includes a separate English-news preparation pipeline for overlap annotation. It can:

* discover up to 5 channel videos in a date range,
* download videos,
* extract 16 kHz mono WAV audio,
* create 480p H.264 proxy videos,
* run WhisperX transcription/alignment/diarization,
* run Pyannote overlapped speech detection,
* merge OSD and diarization-intersection overlap candidates,
* build per-episode JSON,
* chunk proxy videos into 15-minute files,
* export one Label Studio task per chunk.

Example:

```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
python run_english_news_pipeline.py \
  --channel-url "https://www.youtube.com/@CHANNEL/videos" \
  --limit 5 \
  --start-date 2026-06-01 \
  --end-date 2026-06-30
```

For explicit videos:

```bash
python run_english_news_pipeline.py \
  --video-id VIDEO_ID_1 \
  --video-id VIDEO_ID_2
```

The roster extraction and speaker-name linking fields are intentionally left empty until LLM extraction and manual evidence checks are performed.

### V2 Pilot Analysis Layer

A separate, artifact-producing analysis layer runs on top of the v2 `final.json`
files. Each stage is versioned, writes one JSON artifact, and is driven by a
`make` target. Run everything from the repository root with the Python 3.11
virtualenv (`.venv`) active; use `make py` once to create it.

```bash
make py                 # create .venv and install pinned deps (once)
make topics             # Phase 3: topic_modeling.json + topic_windows.json + topic_modeling_review.json
make topic-analysis     # embedding-based topic boundary segmentation sample
make drift              # Phase 5: interruption_topic_drift.json (needs topic_modeling.json)
make discourse          # Phase 4: discourse_summary.json
make discourse-review   # Phase 4: discourse_summary_review.json (needs discourse_summary.json)
make host VIDEO=7dKqZ9ERgiQ REFERENCE=SPEAKER_24 TRIM=720   # Phase 2: host_identification.json
make report             # Phase 6: regenerate PILOT_EPISODE_REPORT.md from all artifacts
make test               # full unit-test suite
```

Artifacts produced (schemas are versioned in each file):

| phase | module | artifact | schema |
| --- | --- | --- | --- |
| 1 | `pilot_episode_audit.json` (authored) | `pilot_episode_audit.json` | `pilot-audit-1.0` |
| 2 | `v2/identity/host.py` + `v2/cli/host.py` | `data/v2/analysis/host_identification.json` | `host-identification-1.0` |
| 3 | `topic_modeling.py` | `topic_modeling.json`, `topic_windows.json`, `topic_modeling_review.json` | `topic-windows-1.0`, `topic-modeling-review-1.0` |
| 4 | `discourse_summary.py` + `discourse_review.py` | `discourse_summary.json`, `discourse_summary_review.json` | `2.0`, `discourse-summary-review-1.0` |
| 5 | `topic_drift.py` | `interruption_topic_drift.json` | `interruption-topic-drift-1.0` |
| 6 | `pilot_report.py` | `PILOT_EPISODE_REPORT.md` | — |

Order matters: `drift` reads `topic_modeling.json`, and `discourse-review` reads
`discourse_summary.json`. Topic labels, detected topic changes, and
interruption-topic associations are candidate outputs, not validated ground
truth; `discourse_review.py` re-derives the summary metrics independently and
reports any mismatch.

## Citation
Please consider citing the following paper when using our code and dataset.

```bibtex
@inproceedings{10.1145/3637528.3671532,
author = {Agarwal, Anmol and Priyadarshi, Pratyush and Sinha, Shiven and Gupta, Shrey and Jangra, Hitkul and Kumaraguru, Ponnurangam and Garimella, Kiran},
title = {Television Discourse Decoded: Comprehensive Multimodal Analytics at Scale},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671532},
doi = {10.1145/3637528.3671532},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4752–4763},
numpages = {12},
keywords = {bias detection, incivil speech, multimodal analysis, television, video analysis},
location = {Barcelona, Spain},
series = {KDD '24}
}
```
