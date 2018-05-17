#!/usr/bin/env python3
"""
    image_processing.py - Preprocessing images (Image resize)
    Author: Dung Le (dungle@bennington.edu)
    Date: 03/29/2018
"""

import cv2
import os
rootdir = './data/ImageNet/101_ObjectCategories/'

''' Resize image to dimension 120px x 120px '''
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file_path != './data/ImageNet/101_ObjectCategories/.DS_Store' and \
           file_path != './data/ImageNet/101_ObjectCategories/BACKGROUND_Google/.DS_Store' and \
           file_path != './data/ImageNet/101_ObjectCategories/Faces/.DS_Store':
            image = cv2.imread(file_path)
            print(file_path)
            dim = (120, 120)
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(file_path, resized)
