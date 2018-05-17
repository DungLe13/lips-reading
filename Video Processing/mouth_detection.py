#!/usr/bin/env python3
"""
    mouth_detection.py - Mouth Detection from video and Capture into frames
    Author: Dung Le (dungle@bennington.edu)
    Date: 03/23/2018
"""

"""
    Note: this code works. However, the current haar cascade file to detect mouth
          is terrible in detecting mouth region.
"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('cascade_files/Mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')

img = cv2.imread('sample-data/frames/frame_47.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    mouth = mouth_cascade.detectMultiScale(roi_gray,2.0,25)
    for (ex,ey,ew,eh) in mouth:
        cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 3)
        crop_img = img[ey:ey+eh, ex:ex+ew]

cv2.imshow('mouth', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
