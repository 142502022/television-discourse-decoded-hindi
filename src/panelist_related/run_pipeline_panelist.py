import os
import sys
import json


CURR_FILE_DIR = os.path.dirname(__file__)
logger = debate_utils.get_logger()

def load_video_ids(args_received):
    if args_received.endwith(".json"):
        assert os.path.exists(args_received)
        with open(args_received) as fd:
            return json.load(fd)
    else:
        return [args_received]
    
vid_id_list = load_video_ids(sys.argv[1])