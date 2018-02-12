from tqdm import tqdm
import formatter as fmt
import csv
import random
import collections
import itertools
import numpy as np
import json

MAIN_DATASET_PATH = "Sentiment Analysis Dataset.csv"
# POS_DATASET_PATH = 'tw-data.pos'
# NEG_DATASET_PATH = 'tw-data.neg'
POS_DATASET_PATH = 'rt-polarity.pos'
NEG_DATASET_PATH = 'rt-polarity.neg'


#Divides the dataset into positive and negative datasets
def create_sets():
    pos_dataset = open("tw-data.pos", "w")
    neg_dataset = open("tw-data.neg", "w")

    with open(MAIN_DATASET_PATH, "r",encoding='UTF-8') as dataset:
        lines = [n for n in dataset]
    print("Creating datasets...")
    e_p =0
    e_n =0
    for line in tqdm(csv.reader(lines), total=len(lines)):
        tweet = line[3].strip()
        tweet = fmt.all(tweet)
        if line[1].strip() == '1':
            try:
                pos_dataset.write(tweet.strip()+'\n')
            except UnicodeEncodeError:
                e_p+=1
        else:
            try:
                neg_dataset.write(tweet.strip()+'\n')
            except UnicodeEncodeError:
                e_n+=1
    print("DONE\nP errors: "+str(e_p)+"\nN errors: "+str(e_n))

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

def load_data():
    positive_text = list(open(POS_DATASET_PATH).readlines())
    positive_text = [s.strip() for s in positive_text]
    negative_text = list(open(NEG_DATASET_PATH).readlines())
    negative_text = [s.strip() for s in negative_text]
    text = positive_text + negative_text
    positive_labels = [[0, 1] for _ in positive_text]
    negative_labels = [[1, 0] for _ in negative_text]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return text, labels

#create_sets()
#build_vocab(0.25)
