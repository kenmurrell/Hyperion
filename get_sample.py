import sys
import random

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
