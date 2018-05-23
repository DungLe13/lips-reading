#!/usr/bin/env python3
"""
    vocab_builder.py - Vocab Builder for GRID Dataset
    Author: Dung Le (dungle@bennington.edu)
    Date: 04/30/2018
"""

import os
import re
import json

def get_vocab():
    path = "../GRID corpus/aligns"
    vocab = []

    for file in os.listdir(path):
        if file != ".DS_Store":
            file_path = path + '/' + file
            with open(file_path, mode='r') as align:
                lines = align.readlines()
                sent = []
                for line in lines:
                    word = re.sub('[^a-zA-Z]+', '', line)
                    if word != 'sil':
                        sent.append(word)
                vocab += sent

    # Vocabulary is a set containing unique words from aligns
    vocabulary = set(vocab)

    """
    The dictionary will take the form:
        {'white': 0, 'p': 1, 'bin': 2, 'soon': 3, 'c': 4, 's': 5, 'seven': 6,
        'h': 7, 'f': 8, 'o': 9, 'b': 10, 'k': 11, 'with': 12, 'zero': 13,
        'at': 14, 'green': 15, 'j': 16, 'n': 17, 'set': 18, ..., 'q': 51}
    """
    dictionary = {}
    count = 0
    for word in vocabulary:
        dictionary[word] = count
        count += 1
    
    return dictionary

v = get_vocab()
with open('vocab.txt', 'w') as outfile:
    json.dump(v, outfile)
