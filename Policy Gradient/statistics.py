#!/usr/bin/env python3
"""
    statistics.py - Perform some statistics on trained data
    Author: Dung Le (dungle@bennington.edu)
    Date: 05/22/2018
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

summary_file = 'files/GRIDLipReading-experiment-1/events.out.tfevents.1527032236.Dungs-MacBook-Pro.local'
loss = []

for event in tf.train.summary_iterator(summary_file):
    for value in event.summary.value:
        if value.tag == 'compute_pg_gradients/total_loss':
            print(value.simple_value)
            loss.append(value.simple_value)

print(len(loss))
train_loss = np.array(loss)
plt.plot(train_loss, label="Train")
plt.show()
