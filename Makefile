# Television Discourse Decoded — V2 pipeline driver.
#
# Reads video IDs from ids.json (a JSON list) and runs the pipeline stages for
# each video. Every stage is idempotent: it skips videos whose artifacts
# already exist, so re-running is safe and cheap.
#
# Usage:
#   make               # full pipeline for every id in ids.json
#   make media         # step 1-3: download + audio + proxy + metadata
#   make asr           # sterx transcrp 4:   whispeiption + diarization
#   make diarize       # step 4b:  speaker diarization
#   make osd           # step 5:   overlap detection + analysis
#   make verify        # step 6:   three-case overlap candidate reconciliation
#   make roster        # step 7:   Gemini roster extraction
#   make identity      # step 5+7: roles + embeddings + roster
#   make status        # per-video artifact presence table
#
# Env:
#   HF_TOKEN      required for diarize/osd/identity (gated pyannote models)
#   GEMINI_API_KEY required for roster
#   VIDEO=ID      limit to a single video, e.g. make asr VIDEO=oFW8Av_G90g

PY    := .venv/bin/python
IDS   := $(shell $(PY) -c "import json;print(' '.join(json.load(open('ids.json'))))" 2>/dev/null)
ifeq ($(strip $(IDS)),)
$(error ids.json not found or not a JSON list; create it as ["id1","id2",...])
endif

VIDEO ?=
ifeq ($(strip $(VIDEO)),)
VIDEOS := $(IDS)
else
VIDEOS := $(VIDEO)
endif

.PHONY: all media asr diarize osd verify roster identity status clean py

## full pipeline (steps 1-10) for all videos
all: media asr diarize osd verify link finalize

## full pipeline through Label Studio (steps 1-12)
all12: media asr diarize osd verify link finalize chunk labelstudio

## steps 1-3: download episode + extract audio + 480p proxy + metadata
media:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.prepare_media --video-id "$$v" || exit 1; \
	done

## step 4: whisperx transcription (word-timed segments)
asr:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.transcribe --video-id "$$v" || exit 1; \
	done

## step 4b: speaker diarization (requires HF_TOKEN)
diarize:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN (gated pyannote models)"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.diarize --video-id "$$v" || exit 1; \
	done

## step 5: overlap detection (OSD) + crosstalk/interruption analysis
osd:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN (gated pyannote models)"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.osd --video-id "$$v" || exit 1; \
	done

## step 6: reconcile OSD vs diarization -> three-case candidate intervals
verify:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.verify --video-id "$$v" || exit 1; \
	done

## step 7: Gemini roster extraction (requires GEMINI_API_KEY)
roster:
	@test -n "$$GEMINI_API_KEY" || { echo "ERROR: set GEMINI_API_KEY"; exit 1; }
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.roster --video-id "$$v" || exit 1; \
	done

## step 8: random evidence-timestamp validation across the corpus
validate:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.validate SAMPLE --video-id "$$v" || exit 1; \
	done
	$(PY) -m v2.cli.validate RUN --data-dir data/v2/videos

## step 9: cluster to real-name linking (roster + roles -> linked.json)
link:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.link --video-id "$$v" || exit 1; \
	done

## step 10: assemble per-episode final.json from every stage
finalize:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.finalize --video-id "$$v" || exit 1; \
	done

## step 11: cut proxy into 15-minute chunks + slice episode JSON
chunk:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.chunk --video-id "$$v" || exit 1; \
	done

## step 12: export chunks as one Label Studio task per chunk (JSONL)
labelstudio:
	@for v in $(VIDEOS); do \
		$(PY) -m v2.cli.labelstudio --video-id "$$v" || exit 1; \
	done

## roles + voice embeddings + named roster
identity:
	@test -n "$$HF_TOKEN" || { echo "ERROR: set HF_TOKEN"; exit 1; }
	@test -n "$$GEMINI_API_KEY" || { echo "ERROR: set GEMINI_API_KEY for step 7"; exit 1; }
	@$(foreach v,$(VIDEOS),$(PY) -m v2.cli.identity --video-id $(v) && $(PY) -m v2.cli.roster --video-id $(v))

## per-video artifact presence table
status:
	@$(PY) -m v2.cli.status

## run the unit test suite
test: py
	$(PY) -m pytest -q

## create venv and install pinned deps
py:
	@test -d .venv || python3.11 -m venv .venv
	./.venv/bin/pip install -q -U pip
	./.venv/bin/pip install -q -r requirements-v2.txt

## delete all generated video artifacts (ids.json preserved)
clean:
	@rm -rf data/v2/videos/* data/* 2>/dev/null; \
	echo "Removed generated artifacts under data/. ids.json kept."