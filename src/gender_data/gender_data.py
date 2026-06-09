import json
import os
from ..config_constants import ConfigConstants
from tv_debs_utils import debate_utils

logger = debate_utils.get_logger()
def load_video_ids(args_received):
    """
    Load video IDs from command line argument or JSON file.

    Args:
        args_received (str): Command line argument (video ID or path to JSON file)

    Returns:
        list: List of video IDs
    """
    if args_received.endswith(".json"):
        assert os.path.exists(args_received)
        with open(args_received) as fd:
            return json.load(fd)
    else:
        return [args_received]

# Load video IDs
vid_id_list = load_video_ids(sys.argv[1])


def run_gender_pipeline(curr_yt_id):
    gender_file_dir = os.path.join(ConfigConstants.GENDER_FILE_DIR, f"{curr_yt_id}.mp4")
    
