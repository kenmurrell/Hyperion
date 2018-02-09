import tensorflow as tf
import numpy as np
import csv
import prep
import os
import time
import random
import sys
import json

X_PATH = 'xfile.npy'
Y_PATH = 'yfile.npy'
VOC_PATH = 'vocab.json'


# Load & shuffle data
print("Loading test data...")
vocabulary = json.load(open(VOC_PATH))
x = np.load(X_PATH)
y = np.load(Y_PATH)
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x[shuffle_indices]
y_train = y[shuffle_indices]
print(y_shuffled)

# Set hyperparameters
learning_rate = 0.0001

# Set placeholders
data_in = tf.placeholder(tf.int32, [None, x_train.shape[1]], name='data_in')
data_out = tf.placeholder(tf.float32, [None, y_train.shape[1]], name='data_out')
keep_prob = tf.placeholder(tf.float32)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# Create model

#
# # Parameters
# filter_sizes = '3,4,5'
# batch_size = 128
# valid_freq = 1
# checkpoint_freq = 1
#
# sequence_length = x_train.shape[1]
# num_classes = y_train.shape[1]
# vocab_size = len(vocabulary)
# filter_sizes = map(int, filter_sizes.split(','))
# validate_every = len(y_train) / (batch_size * valid_freq)
# checkpoint_every = len(y_train) / (batch_size * checkpoint_freq)
