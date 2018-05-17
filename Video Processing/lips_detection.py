#!/usr/bin/env python3
"""
    lips_detection.py - Detection of lips region from image
    Author: Dung Le (dungle@bennington.edu)
    Date: 04/09/2018
"""

import numpy as np
import dlib
import imutils

import cv2
import os
from imutils import face_utils

def mouth_detection(shape_predictor, img, saved_name, saved_path):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
     
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
     
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'mouth':
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
         
                # loop over the subset of facial landmarks, drawing the specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                    # extract the ROI of the face region as a separate image
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = image[y-12:y + h + 12, x-5:x + w + 5]
                    roi = imutils.resize(roi, width=125, inter=cv2.INTER_CUBIC)
         
                    # save the mouth region to a directory
                    cv2.imwrite(os.path.join(saved_path, saved_name), roi)

if __name__ == "__main__":
    frames_dir = '../GRID corpus/frames'
    for subdir, dirs, files in os.walk(frames_dir):
        for file in files:
            if subdir != frames_dir:
                folder_name = subdir[-6:]
                saved_dir = "../GRID corpus/lips/{0}".format(folder_name)
                if not os.path.isdir(saved_dir):
                    os.makedirs(saved_dir)

            file_path = os.path.join(subdir, file)
            if subdir[-6:] >= 'bbbp6p' and file != '.DS_Store':
                mouth_detection('cascade_files/shape_predictor_68_face_landmarks.dat', file_path, file, saved_dir)
