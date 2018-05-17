#!/usr/bin/env python3
"""
    lstm_stacked.py - Recurrent Neural Network with stacked LSTM
    Author: Dung Le (dungle@bennington.edu)
    Date: 05/01/2018
"""

import tensorflow as tf
import numpy as np
import random

import sys
sys.path.append('../CNN/pretrained-VGG16')
import pre_convnet
from pre_convnet import X

class LSTM():
    def __init__(self, embs, state_size, num_actions, num_layers):
        self.embs = embs          # initial input of the LSTM model
        self.state_size = state_size
        self.num_actions = num_actions
        self.num_layers = num_layers

    # build model
    def lstm_model(self):
        # tf.reset_default_graph()
        print(self.embs.shape)
        xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
        rnn_inputs = tf.nn.embedding_lookup(self.embs, xs_)
        print(rnn_inputs.shape)
        
        # initial hidden state
        '''
        init_state = tf.placeholder(shape=[2, self.num_layers, None, self.state_size],
                                    dtype=tf.float64, name='initial_state')
        '''
        init_state = tf.convert_to_tensor(np.zeros([2, self.num_layers, 1000, 1000]),
                                          dtype=tf.float64, name='initial_state')
        
        # initializer
        xav_init = tf.contrib.layers.xavier_initializer
        
        # parameters
        W = tf.get_variable(name='W', shape=[self.num_layers, 4, self.state_size, self.state_size],
                            dtype=tf.float64, initializer=xav_init())
        U = tf.get_variable(name='U', shape=[self.num_layers, 4, self.state_size, self.state_size],
                            dtype=tf.float64, initializer=xav_init())
        b = tf.get_variable(name='b', shape=[self.num_layers, 2, self.state_size],
                            dtype=tf.float64, initializer=tf.constant_initializer(0.))

        ''' Steps in LSTM cells '''
        def step(prev, x):
            # gather previous internal state and output state
            st_1, ct_1 = tf.unstack(prev)
            # iterate through layers
            st, ct = [], []
            inp = x

            for i in range(self.num_layers):
                ''' Four gates of each LSTM cell '''
                # input gate
                ig = tf.sigmoid(tf.matmul(inp, U[i][0]) + tf.matmul(st_1[i], W[i][0]))
                # forget gate
                fg = tf.sigmoid(tf.matmul(inp, U[i][1]) + tf.matmul(st_1[i], W[i][1]))
                # output gate
                og = tf.sigmoid(tf.matmul(inp, U[i][2]) + tf.matmul(st_1[i], W[i][2]))
                # gate weights
                g = tf.tanh(tf.matmul(inp, U[i][3]) + tf.matmul(st_1[i], W[i][3]))

                # new internal cell state
                ct_i = ct_1[i]*fg + g*ig + b[i][0]
                # ouput state
                st_i = tf.tanh(ct_i)*og + b[i][1]
                inp = st_i
                st.append(st_i)
                ct.append(ct_i)

            return tf.stack([st, ct])

        ''' Scan through the RNN inputs '''
        print(tf.transpose(rnn_inputs, [1, 0, 2]).shape)
        states = tf.scan(step, tf.transpose(rnn_inputs, [1, 0, 2]),
                         initializer=init_state)
        # print(states.shape)
        last_state = states[-1]

        ''' Predictions '''
        V = tf.get_variable(name='V', shape=[self.state_size, self.num_actions], dtype=tf.float64,
                            initializer=xav_init())
        bo = tf.get_variable(name='bo', shape=[self.num_actions], dtype=tf.float64,
                             initializer=tf.constant_initializer(0.))

        states = tf.transpose(states, [1, 2, 3, 0, 4])[0][-1]
        # print(states.shape)
        # flatten the states to 2D matrix for matmul with V
        states_reshaped = tf.reshape(states, [1, self.state_size])
        # print(states_reshaped.shape)
        logits = tf.matmul(states_reshaped, V) + bo
        # predictions
        predictions = tf.nn.softmax(logits)
        print(predictions.shape)
        return predictions

    def build(self):
        ''' Build Graph '''
        print("Start building graph for LSTM...")
        return self.lstm_model()

if __name__ == "__main__":
    # sess = tf.Session()
    model = LSTM(embs=X, state_size=1000, num_actions=52, num_layers=2)
    print(model.build())
