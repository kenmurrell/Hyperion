import tensorflow as tf
import numpy as np

# Load data
x, y, vocabulary, vocabulary_inv = load_data(FLAGS.dataset_fraction)

# Randomly shuffle data
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
