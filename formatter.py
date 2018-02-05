import regex as re

def unicodetoascii(sample):
    sample = (sample.
    		replace('\\xe2\\x80\\x99', "'").
            replace('\\xc3\\xa9', 'e').
            replace('\\xe2\\x80\\x90', '-').
            replace('\\xe2\\x80\\x91', '-').
            replace('\\xe2\\x80\\x92', '-').
            replace('\\xe2\\x80\\x93', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x98', "'").
            replace('\\xe2\\x80\\x9b', "'").
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9d', '"').
            replace('\\xe2\\x80\\x9e', '"').
            replace('\\xe2\\x80\\x9f', '"').
            replace('\\xe2\\x80\\xa6', '...').
            replace('\\xe2\\x80\\xb2', "'").
            replace('\\xe2\\x80\\xb3', "'").
            replace('\\xe2\\x80\\xb4', "'").
            replace('\\xe2\\x80\\xb5', "'").
            replace('\\xe2\\x80\\xb6', "'").
            replace('\\xe2\\x80\\xb7', "'").
            replace('\\xe2\\x81\\xba', "+").
            replace('\\xe2\\x81\\xbb', "-").
            replace('\\xe2\\x81\\xbc', "=").
            replace('\\xe2\\x81\\xbd', "(").
            replace('\\xe2\\x81\\xbe', ")")
                 )
    return sample

#removes emojis....why to people use so many goddammn emojis
def removeemojis(sample):
    pattern = re.compile("["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    sample = pattern.sub(r'', sample)
    return sample

#removes links
def removelinks(sample):
    sample = re.sub(r'(https:[/][/]|http:[/][/]|www[.])[a-zA-Z0-9\-\./]*','', sample,flags=re.M)
    return sample

#removes reply prefixes
def removereply(sample):
    sample = re.sub(r'(@)[a-zA-Z0-9\-\.\_\:/]*', '', sample, flags=re.I|re.M)
    sample = re.sub(r'(RT)[\s]*','', sample, flags=re.I|re.M)
    return sample

#cleans the garbage
def clean(sample):
    #remove all illegal characters
    sample = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", sample,flags=re.I)
    #normalizes duplicates, shortens triplicates
    sample = re.sub(r'(.)\1+', r'\1\1', sample, flags=re.I)
    #separates contractions
    sample = re.sub(r"\`", "\'", sample, flags=re.I)
    sample = re.sub(r"\'s", " \'s", sample, flags=re.I)
    sample = re.sub(r"\'ve", " \'ve", sample, flags=re.I)
    sample = re.sub(r"n\'t", " n\'t", sample, flags=re.I)
    sample = re.sub(r"\'re", " \'re", sample, flags=re.I)
    sample = re.sub(r"\'d", " \'d", sample, flags=re.I)
    sample = re.sub(r"\'ll", " \'ll", sample, flags=re.I)
    #separates interrogative/exclamation clause
    sample = re.sub(r"!", " ! ", sample, flags=re.I)
    sample = re.sub(r"\?", " ? ", sample, flags=re.I)
    #remove excess apostrophes
    sample = re.sub(r"(')([^vsmdl])",r"\2", sample, flags=re.I)
    #closes whitespace
    sample = re.sub(r"\s{2,}", " ", sample, flags=re.I)

    return sample.strip().lower()

def all(sample):
    #sample = unicodetoascii(sample)
    sample = removeemojis(sample)
    sample = removelinks(sample)
    sample = removereply(sample)
    sample = clean(sample)
    return sample

#
# print(all('RT @h_webber: @realDonaldTrump #MAGA @realDonaldTrump https://t.co/AL7jwPTQPE this guys sucks, #retard'))
#
