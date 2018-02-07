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
padding_word="<PAD/>"

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def weight_variable(shape, name):
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(var, name=name)

def bias_variable(shape, name):
    var = tf.constant(0.1, shape=shape)
    return tf.Variable(var, name=name)

def evaluate_sentence(sentence, vocabulary):
    x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))
    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    unnorm_result = sess.run(network_out, feed_dict={data_in: x_to_eval,
                                                     dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    print('Custom input evaluation:', network_sentiment)
    print('Actual output:', str(unnorm_result[0]))


option = sys.argv[1]

CHECKPOINT_FILE_PATH = 'ckpt.ckpt'

# Load & shuffle data
print("Loading test data...")
#Sprep.build_vocab(0.25)
vocabulary = json.load(open(VOC_PATH))
x = np.load(X_PATH)
y = np.load(Y_PATH)
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
test_index = int(len(x) * 0.1) #ratio of test data
x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

# Parameters
filter_sizes = '3,4,5'
batch_size = 128
valid_freq = 1
checkpoint_freq = 1

sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(vocabulary)
filter_sizes = map(int, filter_sizes.split(','))
validate_every = len(y_train) / (batch_size * valid_freq)
checkpoint_every = len(y_train) / (batch_size * checkpoint_freq)

#Session
sess = tf.InteractiveSession()
# Set computation device
#device = '/gpu:0'
device = '/cpu:0'
#Network
print("Building network...")
with tf.device(device):
    # Placeholders
    data_in = tf.placeholder(tf.int32, [None, sequence_length], name='data_in')
    data_out = tf.placeholder(tf.float32, [None, num_classes], name='data_out')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    # Stores the accuracy of the model for each batch of the validation testing
    valid_accuracies = tf.placeholder(tf.float32)
    # Stores the loss of the model for each batch of the validation testing
    valid_losses = tf.placeholder(tf.float32)

    #Embedding layer
    embedding_size = 128
    with tf.name_scope('embedding'):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1.0,1.0), name='embedding_matrix')
        embedded_chars = tf.nn.embedding_lookup(W, data_in)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    #Convolution + ReLu + Pooling layers
    num_filters = 128
    pooled_outputs=[]
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            # Convolution Layer
            filter_shape = [filter_size,embedding_size,1,num_filters]
            W = weight_variable(filter_shape, name='W_conv')
            b = bias_variable([num_filters], name='b_conv')
            conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding='VALID',name='conv')
            # Activation function
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # Maxpooling layer
            ksize = [1,sequence_length - filter_size + 1,1,1]
            pooled = tf.nn.max_pool(h,ksize=ksize,strides=[1, 1, 1, 1],padding='VALID',name='pool')
        pooled_outputs.append(pooled)
    # Combine the pooled feature tensors
    num_filters_total = num_filters * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Dropout
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # Output layer
    with tf.name_scope('output'):
        W_out = weight_variable([num_filters_total, num_classes], name='W_out')
        b_out = bias_variable([num_classes], name='b_out')
        network_out = tf.nn.softmax(tf.matmul(h_drop, W_out) + b_out)

    # Loss function
    cross_entropy = -tf.reduce_sum(data_out * tf.log(network_out))

    # Training algorithm
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Testing operations
    correct_prediction = tf.equal(tf.argmax(network_out, 1),
                                  tf.argmax(data_out, 1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Validation ops
    valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
    valid_mean_loss = tf.reduce_mean(valid_losses)

# Init session
if option=='load':
    print('Data processing OK, loading network...')
    saver = tf.train.Saver()
    try:
        saver.restore(sess, CHECKPOINT_FILE_PATH)
    except:
        print('Error!')
        sess.run(tf.global_variables_initializer())
else:
    print('Initializing...')
    sess.run(tf.global_variables_initializer())

#TRAINING
if option=='train':
    # Batches
    print(zip(x_train, y_train))
    epochs =3
    batches = batch_iter(zip(x_train, y_train), batch_size, epochs)
    test_batches = list(batch_iter(zip(x_test, y_test), batch_size, 1))
    my_batch = batches.next()  # To use with human_readable_output()

    # Pretty-printing variables
    global_step = 0
    batches_in_epoch = len(y_train) / batch_size
    batches_in_epoch = batches_in_epoch if batches_in_epoch != 0 else 1
    total_num_step = FLAGS.epochs * batches_in_epoch

    batches_progressbar = tqdm(batches, total=total_num_step,desc='Starting training...')

    for batch in batches_progressbar:
        global_step += 1
        x_batch, y_batch = zip(*batch)

        # Run the training step
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 0.5}
        train_result, loss_summary_result = sess.run([train_step, loss_summary],feed_dict=feed_dict)

        # Print training accuracy
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        current_epoch = (global_step / batches_in_epoch)

        desc = 'Epoch: {} - loss: {:9.5f} - acc: {:7.5f}'.format(current_epoch,current_loss,accuracy_result)
        batches_progressbar.set_description(desc)

        # Write loss summary
        summary_writer.add_summary(loss_summary_result, global_step)

        # Validation testing
        # Evaluate accuracy as (correctly classified samples) / (all samples)
        # For each batch, evaluate the loss
        if global_step % validate_every == 0:
            accuracies = []
            losses = []
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                feed_dict = {data_in: x_test_batch,
                             data_out: y_test_batch,
                             dropout_keep_prob: 1.0}
                accuracy_result = accuracy.eval(feed_dict=feed_dict)
                current_loss = cross_entropy.eval(feed_dict=feed_dict)
                accuracies.append(accuracy_result)
                losses.append(current_loss)

            # Evaluate the mean accuracy of the model using the test accuracies
            mean_accuracy_result, accuracy_summary_result = sess.run(
                [valid_mean_accuracy, valid_accuracy_summary],
                feed_dict={valid_accuracies: accuracies})
            # Evaluate the mean loss of the model using the test losses
            mean_loss_result, loss_summary_result = sess.run(
                [valid_mean_loss, valid_loss_summary],
                feed_dict={valid_losses: losses})

            valid_msg = 'Step %d of %d (epoch %d), validation accuracy: %g, ' \
                        'validation loss: %g' % \
                        (global_step, total_num_step, current_epoch,
                         mean_accuracy_result, mean_loss_result)
            batches_progressbar.write(valid_msg)
            print(valid_msg)  # Write only to file

            # Write summaries
            summary_writer.add_summary(accuracy_summary_result, global_step)
            summary_writer.add_summary(loss_summary_result, global_step)

        if FLAGS.save and global_step % checkpoint_every == 0:
            batches_progressbar.write('Saving checkpoint...')
            print('Saving checkpoint...')
            saver = tf.train.Saver()
            saver.save(sess, CHECKPOINT_FILE_PATH)

    #Final validation testing
    accuracies = []
    losses = []
    for test_batch in test_batches:
        x_test_batch, y_test_batch = zip(*test_batch)
        feed_dict = {data_in: x_test_batch,
                     data_out: y_test_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        accuracies.append(accuracy_result)
        losses.append(current_loss)

    mean_accuracy_result, accuracy_summary_result = sess.run(
        [valid_mean_accuracy, valid_accuracy_summary],
        feed_dict={valid_accuracies: accuracies})
    mean_loss_result, loss_summary_result = sess.run([valid_mean_loss, valid_loss_summary], feed_dict={valid_losses: losses})
    print('End of training, validation accuracy: %g, validation loss: %g' % (mean_accuracy_result, mean_loss_result))

    # Write summaries
    summary_writer.add_summary(accuracy_summary_result, global_step)
    summary_writer.add_summary(loss_summary_result, global_step)

print('Saving checkpoint...')
saver = tf.train.Saver()
saver.save(sess, CHECKPOINT_FILE_PATH)
