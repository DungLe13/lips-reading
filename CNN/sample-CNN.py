#!/usr/bin/env python3
"""
    sample-CNN.py - Example of ConvNet architecture on MNIST dataset
    Author: Dung Le (dungle@bennington.edu)
    Date: 03/24/2018
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/MNIST/', one_hot=True)
print("Size of:")
print("- Training set:\t\t{0}".format(len(data.train.labels)))
print("- Test set:\t\t{0}".format(len(data.test.labels)))
print("- Validation set:\t{0}".format(len(data.validation.labels)))

''' PLACEHOLDER VARIABLES '''
# Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Placeholder variable for true labels
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

''' HELPER FUNCTIONS '''
# 1. Create a new Convolution Layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters)
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution
        layer += biases

        return layer, weights

# 2. Create a new Pooling Layer
def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer

# 3. Create a new ReLU Layer
def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)

        return layer

# 4. Create a new Fully-connected Layer
def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

        # input * weights + biases
        layer = tf.matmul(input, weights) + biases

        return layer

''' Convolutional Neural Network (ConvNet) architecture '''
# Convolutional Layer 1 - Pooling Layer 1 - ReLU Layer 1
layer_conv1, weights_conv1 = new_conv_layer(x_image, 1, 5, 6, "conv1")
layer_pool1 = new_pool_layer(layer_conv1, "pool1")
layer_relu1 = new_relu_layer(layer_pool1, "relu1")

# Convolutional Layer 2 - Pooling Layer 2 - ReLU Layer 2
layer_conv2, weights_conv2 = new_conv_layer(layer_relu1, 6, 5, 16, "conv2")
layer_pool2 = new_pool_layer(layer_conv2, "pool2")
layer_relu2 = new_relu_layer(layer_pool2, "relu2")

# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# Fully-connected Layer 1 - ReLU 3 - Fully-connected Layer 2
layer_fc1 = new_fc_layer(layer_flat, num_features, 128, "fc1")
layer_relu3 = new_relu_layer(layer_fc1, "relu3")
layer_fc2 = new_fc_layer(layer_relu3, 128, 10, "fc2")

# Use SOFTMAX function to normalize the output
with tf.variable_scope('Softmax'):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, 1)

''' Cost Function + Optimizer + Accuracy '''
# Use CROSS ENTROPY cost function
with tf.name_scope('cross_ent'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=layer_fc2)
    cost = tf.reduce_mean(cross_entropy)

# Use ADAM OPTIMIZER
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

# Compute Accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

''' Bookeeping tasks '''
# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

''' TRAINING THE MODEL '''
num_epochs = 3
batch_size = 100
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initializer all variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        train_accuracy = 0

        for batch in range(0, int(len(data.train.labels)/batch_size)):
            # Get a batch of images and labels
            x_batch, y_true_batch = data.train.next_batch(batch_size)
            # Put the batch into a dict for training
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            # Run the optimizer using this batch training data
            sess.run(optimizer, feed_dict=feed_dict_train)

            # Calculate the accuracy on the batch + generate summary and write to file
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)

        train_accuracy /= int(len(data.train.labels)/batch_size)
        # Generate summary and validate the model on the entire validation test
        _, vali_accuracy = sess.run([merged_summary, accuracy],
                                    feed_dict={x:data.validation.images, y_true:data.validation.labels})

        print("Epoch "+str(epoch+1)+" completed")
        print("\tAccuracy:")
        print("\t- Training Accuracy:\t{}".format(train_accuracy))
        print("\t- Validation Accuracy:\t{}".format(vali_accuracy))

        saver.save(sess, './ckpt/mnist-model')
