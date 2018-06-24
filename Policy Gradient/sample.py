#!/usr/bin/env python3
"""
    sample.py - Sample run of REINFORCE using Cart-Pole environment
    Author: Dung Le (dungle@bennington.edu)
    Date: 04/30/2018
"""

from collections import deque

from pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("files/{0}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def policy_network(states):
    # define policy neural network
    W1 = tf.get_variable(name="W1", shape=[state_dim, 20],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable(name="b1", shape=[20], initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)

    W2 = tf.get_variable(name="W2", shape=[20, num_actions],
                         initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable(name="b2", shape=[num_actions], initializer=tf.constant_initializer(0))
    p = tf.matmul(h1, W2) + b2

    return p

pg_reinforce = PolicyGradientREINFORCE(sess, optimizer, policy_network, state_dim, num_actions,
                         summary_writer=writer)

MAX_EPISODES = 10000
MAX_STEPS = 200
episode_history = deque(maxlen=100)

for e in range(MAX_EPISODES):
    # initialize
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        env.render()
        action = pg_reinforce.sampleAction(state[np.newaxis, :])
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        reward = -10 if done else 0.1      # normalize reward
        pg_reinforce.storeRollout(state, action, reward)
        state = next_state

        if done:
            break

    pg_reinforce.updateModel()
    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode {}:".format(e))
    print("- Reward for this episode: {}".format(total_rewards))
    print("- Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, e+1))
        break



