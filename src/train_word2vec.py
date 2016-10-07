import logging
#from nltk import word_tokenize
from gensim.models import word2vec
import re
"""
This is an simple example to demonstrate how to use gensim to train word2vec and store the model.
Please have nltk and gensim installed before running.
Note: This script haven't been tested yet.
Todo:
    1. Regular expression matching
    2. Normaliztion before training
"""

def normalize_wiki(wiki_sentence_path, output_path):
    output_f = open(output_path, "w")
    with open(wiki_sentence_path, "r") as f:
        for i, line in enumerate(f):
	    if i % 500000 == 0:
	        print "Reading Line %d" % (i)
            #replace [string1|string2] with string1
	    while True:
                start = line.find("[")
                middle = line.find("|")
                end = line.find("]")
		if start == -1 or middle == -1 or end == -1:
                    break
		if end <= start:
		    line = line[:end] + line[end+1:]
		    continue
                line = line[:start] + line[start+1: middle] + line[end+1:]
	    s = re.sub(r'[^\w\s]' , "", line.decode("utf-8"), re.UNICODE)
	    output_f.write(s.lower().encode("utf-8"))
    output_f.close()

def read_data(data_path):
    with open(data_path, "r") as f:
	sentences = []
	for i, line in enumerate(f):
	    if i % 500000 == 0:
	        print "Reading Line %d" % (i)
	    sentences.append(line.strip().split())
    return sentences

if __name__ == "__main__":
    #normalize_wiki("../data/small", "../data/normalized_wiki.txt")
    #normalize_wiki("../data/wiki-sentences", "../data/normalized_wiki.txt")
    
    #read data from wiki sentences
    sentences = read_data("../data/normalized_wiki.txt")
	
    #logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
    #train word vectors
    model = word2vec.Word2Vec(sentences, size=300, min_count = 3, workers = 24, window = 5, sg  = 0) # CBOW
    
    #store the model
    model.save('../data/word2vec.mod')

