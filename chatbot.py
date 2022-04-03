#functions required for training 

import nltk

nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)


def stem(word):
    return stemmer.stem(word.lower())


def bagOfWords(token_txt, all):
    token_text = [stem(w) for w in token_txt]
    bag = np.zeros(len(all), dtype=np.float32)

    for i in range(len(all)):
        if all[i] in token_text:
            bag[i] = 1.0

    return bag
