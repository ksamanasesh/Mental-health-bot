import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# nltk.download('punkt_tab')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def words_of_bag(tokenized_sentence,all_words):
    pass


