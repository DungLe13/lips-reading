#!/usr/bin/env python3
"""
    data_transform.py - Transform words into integers for the action spaces
    Author: Dung Le (dungle@bennington.edu)
    Date: 05/01/2018
"""

import os
import re
import json

"""
Using the dictionary from get_vocab function, get alignment transforms
each word into a number between 0 and 51 (since len(vocabulary) = 52)
e.g. bin blue by d eight now  =>  [50, 7, 12, 1, 39, 13]

alignment takes the form:
    [[50, 7, 12, 22, 43, 13], [50, 7, 12, 22, 19, 13],
     [50, 7, 12, 22, 19, 18], [50, 7, 12, 1, 39, 14],
     [50, 7, 12, 1, 39, 13], [50, 7, 12, 1, 39, 16],
     [50, 7, 12, 1, 39, 18], [50, 7, 12, 1, 29, 14],
     [50, 7, 12, 1, 29, 13], [50, 7, 12, 1, 29, 16], ...]
    len(alignment) = 33000 (since there are 33000 videos in GRID corpus)
"""

def get_alignment(vocab):
    alignment = {}
    path = "../GRID corpus/aligns"

    for file in os.listdir(path):
        if file != ".DS_Store":
            file_path = path + '/' + file
            with open(file_path, mode='r') as align:
                lines = align.readlines()
                sent = []
                
                for line in lines:
                    word = re.sub('[^a-zA-Z]+', '', line)
                    if word != 'sil':
                        sent.append(vocab[word])

            alignment[file[:6]] = sent

    return alignment

with open('vocab.txt', 'r') as f:
    vocab = json.load(f)

a = get_alignment(vocab)
with open('alignment.txt', 'w') as outfile:
    json.dump(a, outfile)
