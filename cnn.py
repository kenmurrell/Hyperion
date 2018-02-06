import tensorflow as tf
import numpy as np
import csv
import prep
import os
import times
import random

POS_DATASET_PATH = 'tw-data.pos'
NEG_DATASET_PATH = 'tw-data.neg'

VOC_PATH = 'vocab.csv'
VOC_INV_PATH = 'vocab_inv.csv'
SENTENCE_PATH = 'sentence.csv'
LABEL_PATH = 'label.csv'

# Load data
x, y, vocabulary, vocabulary_inv = prep.build_vocab(0.25)

# Randomly shuffle data
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
test_index = int(len(x) * 0.1) #ratio of test data
x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

# Parameters
sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(vocabulary)
filter_sizes = map(int, '3,4,5'.split(',')) #filter sizes
validate_every = len(y_train) / (128 * 1) #batch sizes, valid_freq
checkpoint_every = len(y_train) / (128 * 1) #batch sizes, checkpoint_freq

# Set computation device
#device = '/gpu:0'
device = '/cpu:0'

#Session
sess = tf.InteractiveSession()
