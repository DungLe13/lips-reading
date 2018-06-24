#!/usr/bin/env python3
"""
    lips_reading_LSTM.py - Lips Reading using Actor-Critic and LSTM as policy network
    Author: Dung Le (dungle@bennington.edu)
    Date: 06/20/2018
"""

from collections import deque

from A3C import ActorCritic
import tensorflow as tf
import numpy as np

import time
import json
import sys
sys.path.append('/Users/danielle13/Desktop/Natural Language Processing/lips-reading/Misc')
import lip_reading_env
from lip_reading_env import GRIDLipReading

with open('/Users/danielle13/Desktop/Natural Language Processing/lips-reading/Misc/video_ids.txt', 'r') as f:
    file_names = json.load(f)

env_name = 'GRIDLipReading'
env = GRIDLipReading()

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("files/{0}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def actor_network(states):
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

def critic_network(states):
    # define weights and biases
    W = tf.get_variable(name="W", shape=[state_dim, 1], dtype=tf.float64,
                       initializer=tf.random_normal_initializer())
    b = tf.get_variable(name="b", shape=[1], dtype=tf.float64,
                       initializer=tf.constant_initializer(0))

    # processing input tensor
    inp = tf.unstack(states, 74, 1)

    # define the LSTM network
    lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(state_dim, forget_bias=1)
    outputs, _ = tf.nn.static_rnn(lstm_layer, inp, dtype=tf.float64)

    p = tf.matmul(outputs[-1], W) + b
    return p

actor_critic = ActorCritic(sess, optimizer, actor_network, critic_network,
                           state_dim, num_actions, summary_writer=writer)

MAX_EPISODES = 5
MAX_STEPS = 6
episode_history = deque(maxlen=100)

for e in range(MAX_EPISODES):
    total_rewards = 0

    for file in file_names[:10]:
        # initialize
        print(file)
        video_rewards = 0
        state = env.reset(file)

        for t in range(MAX_STEPS):
            action, next_state = actor_critic.sampleAction(state[np.newaxis, :])
            print(action)
            reward, done, _ = env.step(action, file)

            video_rewards += reward
            total_rewards += reward

            actor_critic.storeRollout(state, action, reward)
                
            state = np.array(next_state)
            state = tf.Session().run(tf.unstack(state, 1, 1))[0]

            if done: break

    actor_critic.updateModel()

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode {}:".format(e))
    print("- Reward for this episode: {}".format(total_rewards))
    print("- Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
