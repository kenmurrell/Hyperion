import tensorflow as tf
import numpy as np
import os
import time
import prep
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Data Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS

eval_data = "new_dataset.csv"

# Map data into vocabulary
x_raw = prep.load_test_data(eval_data)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# Evaluating
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint((FLAGS.checkpoint_dir))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        #Generates a generator containing batches of training data
        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print results
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
sentiments = np.average(all_predictions, axis=0)*100
print("Positive sentiments: "+str(sentiments)+"%")
print("Saving evaluation under {0}".format((FLAGS.checkpoint_dir)))
with open(os.path.join(FLAGS.checkpoint_dir, "prediction.csv"), 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
