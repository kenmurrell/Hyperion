import sys
import random
import prep
import numpy
import json

def clean_tweets():
    POS_DATASET_PATH = 'tw-data.pos'
    NEG_DATASET_PATH = 'tw-data.neg'
    positive_examples = list(open(POS_DATASET_PATH).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = random.sample(positive_examples, 5)
    negative_examples = list(open(NEG_DATASET_PATH).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = random.sample(negative_examples, 5)
    print('---POSITIVE:---')
    for tweet in positive_examples:
        print(tweet)
    print('\n---NEGATIVE:---')
    for tweet in negative_examples:
        print(tweet)

def vocab():
    prep.build_vocab(0.1)
    print(numpy.load('xfile.npy'))
    print(numpy.load('yfile.npy'))
    with open('vocab.json','r') as vfile :
        print(json.load(vfile))

vocab()
