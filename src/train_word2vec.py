import logging
from nltk import word_tokenize
from gensim import word2vec

"""
This is an simple example to demonstrate how to use gensim to train word2vec and store the model.
Please have nltk and gensim installed before running.
Note: This script haven't been tested yet.
Todo:
    1. Regular expression matching
    2. Normaliztion before training
"""

def read_data(wiki_sentence_path):
    sentences = []
    with open(wiki_sentence_path, "r") as f:
        for line in f:
            #replace [string1|string2] with string1
            while True:
                start = line.find("[")
                middle = line.find("|")
                end = line.find("]")
                if start == -1 or middle == -1 or end == -1:
                    break
                line = line[:start] + line[start+1: middle] + line[end+1:]
            sentences.append(word_tokenize(line))
        return sentences

if __name__ == "__main__":
    #read data from wiki sentences
    sentences = read_data("../data/wiki-sentences")
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    #train word vectors
    model = word2vec.Word2Vec(sentences, size=200)
    
    #store the model
    model.save('../data/word2vec.mod')

