#!/usr/bin/env python3
"""
    vid2frames.py - Convert video to frames using OpenCV
    Author: Dung Le (dungle@bennington.edu)
    Date: 03/23/2018
"""

import cv2
import os, glob

def toFrames(vid, saved_path):
    vidcap = cv2.VideoCapture(vid)
    success, image = vidcap.read()
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(saved_path + '/frame_%d.jpg' % count, image)
        count += 1

    if not success:
        count -= 1
        os.remove(saved_path + '/frame_%d.jpg' % count)

if __name__ == "__main__":
    video_dir = "../GRID corpus/videos"
    for file in os.listdir(video_dir):
        if file.endswith('.mpg') and file[:6] >= 'lbii7s':
            saved_dir = "../GRID corpus/frames/{0}".format(file[:6])
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)

            video = video_dir + '/' + file
            toFrames(video, saved_dir)
