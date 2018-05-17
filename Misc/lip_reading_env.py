#!/usr/bin/env python3
"""
    lip_reading_env.py - Lip Reading Environment using OpenAI Gym
    Author: Dung Le (dungle@bennington.edu)
    Date: 04/30/2018
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
#from data_transform import get_alignment
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
'''
import sys
sys.path.append('../CNN/pretrained-VGG16')
import pre_convnet
from pre_convnet import X
'''
class GRIDLipReading(gym.Env):
    """
    The goal of Lip Reading is to generate text based on the viseme of words.

    After each step, the agent selects a word from the vocabulary. In this case,
    the action space contains all the words in the vocabulary. TODO: vocab_builder -
    extracts all labeled text and put them into the vocabulary.

    The rewards is calculated after the whole sentence is generated using BLEU:
    - reference = labeled sequence
    - candidate = predicted sentence
    - reward = BLEU score between reference and candidate, considering up to 4-grams
    """
    def __init__(self):
        l = np.zeros((1000,), dtype=int)
        h = np.full((1000,), 51)
        
        self.action_space = spaces.Discrete(52)      # len(vocabulary) = 52
        self.observation_space = spaces.Box(low=l, high=h)
        # labeled text for each video
        with open('/Users/danielle13/Desktop/Natural Language Processing/lip-reading/Misc/alignment.txt', 'r') as f:
            self.alignment = json.load(f)

        self.inp = None
        self.observation = np.zeros((6,), dtype=int)
        self.guess_count = 0
        self.guess_max = 5         # since there are only 6 words for one video

        self.seed()
        #self.reset(file)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, file):
        assert self.action_space.contains(action)

        self.observation[self.guess_count] = action
        # compute the BLEU score (reward) between observation and alignment
        # at the end of sentence
        if self.guess_count == 5:
            smoother = SmoothingFunction()
            reward = sentence_bleu([self.alignment[file]], self.observation,
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoother.method1)
        else:
            reward = 0

        self.guess_count += 1
        # if guess_count == 6, then done with 1 video
        done = self.guess_count > self.guess_max

        return reward, done, {"guesses": self.guess_count}

    def reset(self, file):
        self.guess_count = 0
        self.inp = np.load("/Users/danielle13/Desktop/Natural Language Processing/lip-reading/GRID corpus/vectors/{0}.npz".format(file))
        self.observation = np.zeros((6,), dtype=int)
        return self.inp['X']
