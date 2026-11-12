# Five-stage Hindi TV debate analysis pipeline.
# Source of truth: analysis/run_variant_pipeline.py and
# analysis/build_five_stage_comparison.py, not the legacy Make targets.

PYTHON ?= .venv/bin/python
DATA_DIR := data/v2/videos
IDS_FILE := ids.json
LOG_DIR := logs
VIDEO_ID ?=
OVERWRITE ?= 0
OVERWRITE_FLAG := $(if $(filter 1 yes true,$(OVERWRITE)),--overwrite,)
IDS := $(shell $(PYTHON) -c 'import json; print(" ".join(x for x in json.load(open("$(IDS_FILE)")) if isinstance(x, str) and x.strip()))')

.DEFAULT_GOAL := help
.PHONY: help list check original baseline threshold-0.5 threshold-1.0 threshold-2.0 thresholds compare run all clean-generated summary

help:
	@echo "Five-stage Hindi TV debate analysis"
	@echo "  make list"
	@echo "  make check [VIDEO_ID=<id>]"
	@echo "  make original|baseline|threshold-0.5|threshold-1.0|threshold-2.0 VIDEO_ID=<id>"
	@echo "  make run VIDEO_ID=<id> [OVERWRITE=1]"
	@echo "  make all [OVERWRITE=1]"
	@echo "  make compare VIDEO_ID=<id>"
	@echo "OVERWRITE=1 replaces generated baseline/threshold directories only; it never writes root final.json or source ASR/OSD artifacts."

list:
	@$(PYTHON) -c 'import json; [print(v) for v in json.load(open("$(IDS_FILE)")) if isinstance(v, str) and v.strip()]'

check:
	@$(PYTHON) -c 'import json, pathlib, sys; ids=["$(VIDEO_ID)"] if "$(VIDEO_ID)" else json.load(open("$(IDS_FILE)")); missing=[v for v in ids if not (pathlib.Path("$(DATA_DIR)")/v/"final.json").is_file()]; print("Checking", len(ids), "video(s)"); print("Missing final.json:", ", ".join(missing) if missing else "none"); sys.exit(bool(missing))'
	@test -f analysis/run_variant_pipeline.py
	@test -f analysis/build_five_stage_comparison.py

# S1 is the immutable root final.json plus its root analysis artifacts.
original: check
	@echo "S1 source validation complete; original final.json is read-only."

# S2 only. --thresholds with no values asks the runner for baseline only.
baseline: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/baseline.log

# Each threshold starts from root source artifacts; no threshold reads another threshold.
threshold-0.5: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds 0.5 $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/threshold_0.5.log

threshold-1.0: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds 1.0 $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/threshold_1.0.log

threshold-2.0: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds 2.0 $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/threshold_2.0.log

thresholds: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds 0.5 1.0 2.0 $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/thresholds.log

compare: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/build_five_stage_comparison.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/comparison.log
	@test -s $(DATA_DIR)/$(VIDEO_ID)/comparison/five_stage_comparison.json

run: check
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>"; exit 2)
	@mkdir -p $(LOG_DIR)/$(VIDEO_ID)
	@$(PYTHON) analysis/run_variant_pipeline.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) --thresholds 0.5 1.0 2.0 $(OVERWRITE_FLAG) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/run.log
	@$(PYTHON) analysis/build_five_stage_comparison.py --data-dir $(DATA_DIR) --video-id $(VIDEO_ID) 2>&1 | tee $(LOG_DIR)/$(VIDEO_ID)/comparison.log
	@test -s $(DATA_DIR)/$(VIDEO_ID)/comparison/five_stage_comparison.json

all: check
	@set -e; total=0; ok=0; failed=''; for id in $(IDS); do total=$$((total+1)); echo "[$$total] $$id"; if $(MAKE) --no-print-directory run VIDEO_ID=$$id OVERWRITE=$(OVERWRITE); then ok=$$((ok+1)); else failed="$$failed $$id"; fi; done; echo "Completed $$ok/$$total; failed:$${failed:- none}"; test -z "$$failed"

summary:
	@$(PYTHON) -c 'import json, pathlib, datetime; ids=json.load(open("$(IDS_FILE)")); rows=[{"video_id":v,"comparison_exists":(pathlib.Path("$(DATA_DIR)")/v/"comparison/five_stage_comparison.json").is_file()} for v in ids]; pathlib.Path("$(LOG_DIR)").mkdir(exist_ok=True); json.dump({"generated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"videos":rows},open("$(LOG_DIR)/pipeline_summary.json","w"),indent=2); print("Wrote $(LOG_DIR)/pipeline_summary.json")'

clean-generated:
	@test -n "$(VIDEO_ID)" || (echo "ERROR: specify VIDEO_ID=<id>; refusing broad cleanup"; exit 2)
	@rm -rf $(DATA_DIR)/$(VIDEO_ID)/baseline $(DATA_DIR)/$(VIDEO_ID)/threshold_0.5 $(DATA_DIR)/$(VIDEO_ID)/threshold_1.0 $(DATA_DIR)/$(VIDEO_ID)/threshold_2.0 $(DATA_DIR)/$(VIDEO_ID)/comparison $(LOG_DIR)/$(VIDEO_ID)
	@echo "Removed generated five-stage outputs for $(VIDEO_ID); source final.json and source artifacts were preserved."
