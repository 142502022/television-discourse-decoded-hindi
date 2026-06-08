
import json
import os
from time import sleep
from multiprocessing import Process
from detoxify import Detoxify
from ..config_constants import ConfigConstants
from ..tv_debs_utils import debate_utils

# get a logger to use
logger = debate_utils.get_logger()
logger.info("INFO IS FROM THE LOGGER/")

# Detoxify label -> Perspective API attribute mapping
# (keeps output format identical to what the rest of the pipeline expects)
DETOXIFY_TO_PERSPECTIVE = {
    "toxicity":             "TOXICITY",
    "severe_toxicity":      "SEVERE_TOXICITY",
    "identity_attack":      "IDENTITY_ATTACK",
    "threat":               "THREAT",
    "insult":               "INSULT",
    "obscene":              "PROFANITY",   # closest equivalent
}


def build_perspective_style_response(text: str, scores: dict) -> dict:
    """
    Converts Detoxify scores into a dict that mirrors the Perspective API
    response structure, so the rest of the pipeline works unchanged.
    """
    attribute_scores = {}
    for detox_key, persp_key in DETOXIFY_TO_PERSPECTIVE.items():
        score = float(scores.get(detox_key, 0.0))
        attribute_scores[persp_key] = {
            "summaryScore": {
                "value": score,
                "type": "PROBABILITY"
            },
            "spanScores": [
                {
                    "begin": 0,
                    "end": len(text),
                    "score": {
                        "value": score,
                        "type": "PROBABILITY"
                    }
                }
            ]
        }
    return {
        "attributeScores": attribute_scores,
        "languages": ["en"],
        "detectedLanguages": ["en"]
    }


def process_target(files):
    logger.info("Loading Detoxify model (original)...")
    model = Detoxify('original')
    logger.info("Detoxify model loaded.")

    for file_id, file in enumerate(files):
        logger.info(f"Processing: [{file_id}/{len(files)}]: {file}")

        with open(os.path.join(ConfigConstants.TRANSCRIPT_FILE_DIR, f"{file}.json"), 'r') as f:
            transcript_data = json.load(f)

        write_path = os.path.join(
            ConfigConstants.PERSPECTIVE_FILE_DIR, f"{file}.json")

        if os.path.exists(write_path):
            logger.debug(f"Perspective data already exists for: {file}")
            continue

        logger.debug(f"Number of utterances is: {len(transcript_data)}")

        for ind, utterance in enumerate(transcript_data):
            logger.debug(f"Processing utterance: {ind}/{len(transcript_data)}")

            if "perspective" in utterance:
                continue

            if len(utterance["text"]):
                try:
                    scores = model.predict(utterance["text"])
                    response = build_perspective_style_response(
                        utterance["text"], scores)
                    transcript_data[ind]["perspective"] = response
                except Exception as e:
                    logger.exception(
                        f"Error occurred for: {file}, {ind}: {e}")
            else:
                transcript_data[ind]["perspective"] = {}

        with open(write_path, 'w') as f:
            json.dump(transcript_data, f, indent=1)

        sleep(0.5)


files = os.listdir(ConfigConstants.TRANSCRIPT_FILE_DIR)
logger.info(f"Number of files with transcripts: {len(files)}")

files_done = os.listdir(ConfigConstants.PERSPECTIVE_FILE_DIR)
files = sorted(list(set(files).difference(set(files_done))))
logger.info(
    f"Number of files after removing files that are already processed: {len(files)}")

files = list(filter(lambda x: "json" in x, files))
logger.info(f"Number of files after restricting to json files only: {len(files)}")

# Split files across CPU workers (no API keys needed anymore)
NUM_WORKERS = max(1, os.cpu_count() // 2)
process_files = {id: [] for id in range(NUM_WORKERS)}
[process_files[idx % NUM_WORKERS].append(file_name.split('.json')[0])
 for idx, file_name in enumerate(files)]


def run():
    procs = []
    for pID in range(NUM_WORKERS):
        proc = Process(target=process_target, args=(process_files[pID],))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


run()
