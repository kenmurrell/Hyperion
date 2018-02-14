from tqdm import tqdm
import formatter as fmt
import csv
import random
import collections
import itertools
import numpy as np
import json

MAIN_DATASET_PATH = "./datasets/RottenTomatoesSentiments.csv"
POS_DATASET_PATH = './datasets/rt-polarity.pos'
NEG_DATASET_PATH = './datasets/rt-polarity.neg'

#Divides the dataset into positive and negative datasets
def create_sets():
    print("Creating datasets...")
    with open(MAIN_DATASET_PATH, "r",encoding='UTF-8') as dataset, open(POS_DATASET_PATH, "w") as pos, open(NEG_DATASET_PATH, "w") as neg:
        e_p =0
        e_n =0
        dataset_reader = csv.reader(dataset, delimiter=',')
        tweets = list(dataset_reader)
        for tweet in tqdm(tweets):
            tweet[1] = tweet[1].strip()
            tweet[1] = fmt.all(tweet[1])
            if tweet[0].strip()=='1':
                try:
                    pos.write(tweet[1].strip()+'\n')
                except Exception:
                    e_p+=1
            else:
                try:
                    neg.write(tweet[1].strip()+'\n')
                except Exception:
                    e_n+=1
        print("DONE\nP errors: "+str(e_p)+"\nN errors: "+str(e_n))

#Loads training data from their datasets
def load_training_data():
    positive_text = list(open(POS_DATASET_PATH).readlines())
    positive_text = [s.strip() for s in positive_text]
    negative_text = list(open(NEG_DATASET_PATH).readlines())
    negative_text = [s.strip() for s in negative_text]
    text = positive_text + negative_text
    positive_labels = [[0, 1] for _ in positive_text]
    negative_labels = [[1, 0] for _ in negative_text]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return text, labels

#Loads test data from thei datasets
def load_test_data(filename):
    print("Reading data...")
    out=[]
    with open(filename, 'r', encoding='ISO-8859-1') as dataset:
        dataset_reader = csv.reader(dataset, delimiter=',')
        tweets = list(dataset_reader)
        for tweet in tqdm(tweets):
            tweet[0] = tweet[0].strip()
            tweet[0] = fmt.all(tweet[0])
            out.append(tweet[0])
    return out;
