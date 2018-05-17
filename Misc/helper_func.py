#!/usr/bin/env python3
"""
    helper_func.py - List of all helper functions
    Author: Dung Le (dungle@bennington.edu)
    Date: 05/01/2018
"""

import os
import re
import json

def get_video_id():
    video_ids = []
    path = "../GRID corpus/aligns"

    for file in os.listdir(path):
        if file != ".DS_Store":
            video_ids.append(file[:6])

    return video_ids

ids = get_video_id()
with open('video_ids.txt', 'w') as outfile:
    json.dump(ids, outfile)
