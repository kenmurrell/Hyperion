from tqdm import tqdm
import formatter as fmt
import csv
import random
import collections
import itertools
import numpy
import json

MAIN_DATASET_PATH = "Sentiment Analysis Dataset.csv"
POS_DATASET_PATH = 'tw-data.pos'
NEG_DATASET_PATH = 'tw-data.neg'

X_PATH = 'xfile'
Y_PATH = 'yfile'
VOC_PATH = 'vocab.json'
padding_word="<PAD/>"


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
def build_vocab(dataset_fraction):
    positive_examples = list(open(POS_DATASET_PATH).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(NEG_DATASET_PATH).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    print("Generating vocab...")
    positive_examples = random.sample(positive_examples, int(len(positive_examples) * dataset_fraction)) #get a sample dataset from pos
    negative_examples = random.sample(negative_examples, int(len(negative_examples) * dataset_fraction)) #get a sample dataset from neg
    examples = positive_examples + negative_examples
    examples = [s.split(" ") for s in examples]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = numpy.concatenate([positive_labels, negative_labels], 0)
    #examples is a list of all tweets. Each tweet is stored as a list of words (sans spaces)
    #labels is a list of "positive" and "negative" boolean values, whose index corresponds to examples

    length = max(len(x) for x in examples)
    padded_sentences = []
    for i in range(len(examples)):
        sentence = examples[i]
        new_sentence = sentence + [padding_word] * (length - len(sentence))
        padded_sentences.append(new_sentence)
    examples=padded_sentences
    #the tweets stored in examples are all the same length

    ctr = collections.Counter(itertools.chain(*examples))
    vocabulary = {x[0]:i for i,x in enumerate(ctr.most_common())}
    #vocabulary is a dict of all words, with a corresponding value
    x = numpy.array([[vocabulary[word] for word in sentence] for sentence in examples])
    y = numpy.array(labels)
    numpy.save(X_PATH,x)
    numpy.save(Y_PATH,y)
    with open(VOC_PATH,'w') as vfile :
        json.dump(vocabulary, vfile, indent=4)

#create_sets()
#build_vocab(0.25)
