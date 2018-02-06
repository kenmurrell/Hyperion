import tensorflow as tf
import numpy as np

POS_DATASET_PATH = 'tw-data.pos'
NEG_DATASET_PATH = 'tw-data.neg'
VOC_PATH = 'vocab.csv'
VOC_INV_PATH = 'vocab_inv.csv'

build

# Load data
x, y, vocabulary, vocabulary_inv = load_data(FLAGS.dataset_fraction)

# Randomly shuffle data
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

def prep_data():
    voc = csv.reader(open(VOC_PATH))
    voc_inv = csv.reader(open(VOC_INV_PATH))
    vocabulary_inv = [x for x in voc_inv]
    vocabulary = {x: i for x, i in voc}
