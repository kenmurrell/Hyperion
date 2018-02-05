from tqdm import tqdm
import formatter as fmt
import csv
import random
from collections import Counter
import itertools
import numpy

MAIN_DATASET_PATH = "Sentiment Analysis Dataset.csv"
POS_DATASET_PATH = 'tw-data.pos'
NEG_DATASET_PATH = 'tw-data.neg'
VOC_PATH = 'vocab.csv'
VOC_INV_PATH = 'vocab_inv.csv'


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

#Builds the vocabulary tree
def build_vocab(dataset_fraction,padding_word="<PAD/>"):
    positive_examples = list(open(POS_DATASET_PATH).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(NEG_DATASET_PATH).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    print("Generating vocab...")
    positive_examples = random.sample(positive_examples, int(len(positive_examples) * dataset_fraction)) #get a sample dataset from pos
    negative_examples = random.sample(negative_examples, int(len(negative_examples) * dataset_fraction)) #get a sample dataset from neg
    tweets = positive_examples + negative_examples
    tweets = [s.split(" ") for s in tweets]

    print("Generating labels...")
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = numpy.concatenate([positive_labels, negative_labels], 0)

    print("Padding sentences...")
    length = max(len(x) for x in tweets)
    padded_sentences = []
    for i in range(len(tweets)):
        sentence = tweets[i]
        num_padding = length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    sentences=padded_sentences

    print("Vector mapping...")
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("Writing to file...")
    voc = csv.writer(open(VOC_PATH, 'w'))
    voc_inv = csv.writer(open(VOC_INV_PATH, 'w'))
    for key, val in vocabulary.items():
        voc.writerow([key, val])
    for val in vocabulary_inv:
        voc_inv.writerow([val])


create_sets()
build_vocab(0.25)
