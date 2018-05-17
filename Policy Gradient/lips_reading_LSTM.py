#!/usr/bin/env python3
"""
    lips_reading_LSTM.py - Lips Reading using REINFORCE and LSTM as policy network
    Author: Dung Le (dungle@bennington.edu)
    Date: 05/02/2018
"""

from collections import deque

from REINFORCE import REINFORCE
#from lstm_stacked import LSTM, X
import tensorflow as tf
import numpy as np

import json
import sys
sys.path.append('/Users/danielle13/Desktop/Natural Language Processing/lip-reading/Misc')
import lip_reading_env
from lip_reading_env import GRIDLipReading

with open('/Users/danielle13/Desktop/Natural Language Processing/lip-reading/Misc/video_ids.txt', 'r') as f:
    file_names = json.load(f)

env_name = 'GRIDLipReading'
env = GRIDLipReading()

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("files/{0}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def policy_net(states):
    '''
    TODO: self-defined LSTM model
    m = LSTM(embs=states, state_size=1000, num_actions=52, num_layers=2)
    return m.build()
    '''
    # define weights and biases
    W = tf.get_variable(name="W", shape=[state_dim, num_actions], dtype=tf.float64,
                       initializer=tf.random_normal_initializer())
    b = tf.get_variable(name="b", shape=[num_actions], dtype=tf.float64,
                       initializer=tf.constant_initializer(0))

    # processing input tensor
    inp = tf.unstack(states, 74, 1)

    # define the LSTM network
    lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(state_dim, forget_bias=1)
    outputs, _ = tf.nn.static_rnn(lstm_layer, inp, dtype=tf.float64)

    p = tf.matmul(outputs[-1], W) + b
    return p, outputs

pg_reinforce = REINFORCE(sess, optimizer, policy_net, state_dim, num_actions,
                         summary_writer=writer)

MAX_EPISODES = 3
MAX_STEPS = 6
episode_history = deque(maxlen=100)

for e in range(MAX_EPISODES):

    for file in file_names[:5]:
        # initialize
        state = env.reset(file)
        total_rewards = 0

        for t in range(MAX_STEPS):
            action, next_state = pg_reinforce.sampleAction(state[np.newaxis, :])
            print(action)
            reward, done, _ = env.step(action, file)

            total_rewards += reward
            #reward = -1 if done else 0        # normalize reward
            pg_reinforce.storeRollout(state, action, reward)
            state = np.array(next_state)
            state = tf.Session().run(tf.unstack(state, 1, 1))[0]

            if done:
                break

        pg_reinforce.updateModel()
        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}:".format(e))
        print("- Reward for this episode: {}".format(total_rewards))
        print("- Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
