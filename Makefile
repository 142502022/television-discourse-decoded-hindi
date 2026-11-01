
# Television Discourse Decoded — V2 + MTP Analysis Pipeline

PY    := .venv/bin/python
IDS   := $(shell $(PY) -c "import json;print(' '.join(json.load(open('ids.json'))))" 2>/dev/null)

ifeq ($(strip $(IDS)),)
$(error ids.json not found or not a JSON list; expected ["id1","id2",...])
endif

VIDEO ?=
ifeq ($(strip $(VIDEO)),)
VIDEOS := $(IDS)
else
VIDEOS := $(VIDEO)
endif

ANALYSIS_DIR := analysis
VIDEOS_DIR   := data/v2/videos

.PHONY: all all12 media asr diarize osd verify roster identity \
        validate link finalize chunk labelstudio status \
        analysis analysis-followup analysis-classification \
        analysis-turn-gap analysis-clusters analysis-final \
        analysis-numeric topics topic-analysis drift discourse \
        discourse-review report host test py clean

# ============================================================
# FULL PIPELINE
# ============================================================

# Run the entire pipeline for all videos in ids.json
all: media asr diarize osd verify link finalize analysis

# Run the pipeline for one video:
# make all VIDEO=VIDEO_ID

# Original pipeline through Label Studio
all12: media asr diarize osd verify link finalize chunk labelstudio


# ============================================================
# V2 CORE PIPELINE
# ============================================================

# Step 1: Download video + metadata
media:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.prepare_media --video-id "$$v" || exit 1; \
	done

# Step 2: ASR transcription
asr:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.transcribe --video-id "$$v" || exit 1; \
	done

# Step 3: Speaker diarization
diarize:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.diarize --video-id "$$v" || exit 1; \
	done

# Step 4: Overlapping Speech Detection
osd:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.osd --video-id "$$v" || exit 1; \
	done

# Step 5: Reconcile OSD and diarization
verify:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.verify --video-id "$$v" || exit 1; \
	done

# Step 6: Gemini roster extraction
roster:
	@test -n "$$GEMINI_API_KEY" || { echo "ERROR: set GEMINI_API_KEY"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.roster --video-id "$$v" || exit 1; \
	done

# Step 7: Validate artifacts
validate:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.validate SAMPLE --video-id "$$v" || exit 1; \
	done
	$(PY) -m v2.cli.validate RUN --data-dir $(VIDEOS_DIR)

# Step 8: Link speaker clusters to real names
link:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.link --video-id "$$v" || exit 1; \
	done

# Step 9: Generate final.json
finalize:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.finalize --video-id "$$v" || exit 1; \
	done


# ============================================================
# LABEL STUDIO PIPELINE
# ============================================================

chunk:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.chunk --video-id "$$v" || exit 1; \
	done

labelstudio:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.labelstudio --video-id "$$v" || exit 1; \
	done


# ============================================================
# IDENTITY PIPELINE
# ============================================================

identity:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN"; exit 1; }
	@test -n "$$GEMINI_API_KEY" || { echo "ERROR: set GEMINI_API_KEY"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.identity --video-id "$$v" || exit 1; \
		$(PY) -m v2.cli.roster --video-id "$$v" || exit 1; \
	done


# ============================================================
# MTP ANALYSIS PIPELINE
# All analysis scripts are stored in analysis/
# Each script processes one video at a time.
# ============================================================

# Run all the analysis stages sequentially
analysis: analysis-followup \
          analysis-classification \
          analysis-turn-gap \
          analysis-clusters \
          analysis-final \
          analysis-numeric


# Step 10: Overlap follow-up analysis
analysis-followup: finalize
	@echo "===== Overlap Follow-up Analysis ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/overlap_followup_analysis.py \
			--video-id "$$v" || exit 1; \
	done


# Step 11: Overlap follow-up classification
analysis-classification: analysis-followup
	@echo "===== Overlap Follow-up Classification ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/build_overlap_followup.py \
			--video-id "$$v" || exit 1; \
	done


# Step 12: Turn gap analysis
analysis-turn-gap: finalize
	@echo "===== Turn Gap Analysis ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/turn_gap_analysis.py \
			--video-id "$$v" || exit 1; \
	done


# Step 13: Overlap cluster analysis
analysis-clusters: analysis-classification analysis-turn-gap
	@echo "===== Overlap Cluster Analysis ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/overlap_cluster_analysis.py \
			--video-id "$$v" || exit 1; \
	done


# Step 14: Generate final_result.json
analysis-final: analysis-clusters
	@echo "===== Generating final_result.json ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/final_result.py \
			--video-id "$$v" || exit 1; \
	done


# Step 15: Generate numerical-only JSON for visualization
analysis-numeric: analysis-final
	@echo "===== Generating final_result_numeric.json ====="
	@for v in $(VIDEOS); do \
		$(PY) $(ANALYSIS_DIR)/final_result_numeric.py \
			--video-id "$$v" --overwrite || exit 1; \
	done


# ============================================================
# EXISTING TOPIC / DISCOURSE ANALYSIS
# ============================================================

topics:
	$(PY) topic_modeling.py --trim-config data/v2/topic_trims.json \
		--output data/v2/topic_analysis/topic_modeling.json \
		--windows-output data/v2/topic_analysis/topic_windows.json \
		--review-output data/v2/topic_analysis/topic_modeling_review.json

topic-analysis:
	$(PY) topic_analysis.py --all \
		--output data/v2/topic_analysis/topic_analysis_sample.json

drift:
	$(PY) topic_drift.py \
		--topic-modeling data/v2/topic_analysis/topic_modeling.json \
		--output data/v2/topic_analysis/interruption_topic_drift.json

discourse:
	$(PY) discourse_summary.py --videos-dir $(VIDEOS_DIR) \
		--output data/v2/analysis/discourse_summary.json

discourse-review:
	$(PY) discourse_review.py --videos-dir $(VIDEOS_DIR) \
		--summary data/v2/analysis/discourse_summary.json \
		--output data/v2/analysis/discourse_summary_review.json

report:
	$(PY) pilot_report.py --output PILOT_EPISODE_REPORT.md


# ============================================================
# HOST IDENTIFICATION
# ============================================================

REFERENCE ?= SPEAKER_24
TRIM      ?= 720

host:
	@test -n "$(VIDEO)" || { echo "ERROR: set VIDEO=<id>"; exit 1; }
	$(PY) -m v2.cli.host --video-id "$(VIDEO)" \
		--reference-label "$(REFERENCE)" \
		--trim-start "$(TRIM)" \
		--output data/v2/analysis/host_identification.json


# ============================================================
# UTILITIES
# ============================================================

status:
	@$(PY) -m v2.cli.status

test: py
	$(PY) -m pytest -q

py:
	@test -d .venv || python3.11 -m venv .venv
	./.venv/bin/pip install -q -U pip
	./.venv/bin/pip install -q -r requirements-v2.txt

clean:
	@rm -rf data/v2/videos/* data/* 2>/dev/null; \
	echo "Removed generated artifacts under data/. ids.json kept."
