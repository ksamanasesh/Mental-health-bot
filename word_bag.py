import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np
# nltk.download('punkt_tab')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def words_of_bag(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bog = np.zeros(len(all_words), dtype=np.float32)

    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bog[idx] = 1.0
        
    return bog  