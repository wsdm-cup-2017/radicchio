import sys
from gensim.models import word2vec
from utils import *

def all_words(terms):
	S = set()
	for term in terms:
		for word in term:
			S.add(word)
		S.add("_".join(term))
	return S

def reduce_w2v(w2v, words, output_path):
	with open(output_path, "w") as f:
		f.write("%d %d\n" % (len(words), 300))
		for word in sorted(list(words)):
			if word in w2v.vocab:
				f.write(word+" ")
				vector = map(lambda x : "{0:.6f}".format(x), w2v[word])
				f.write(" ".join(vector) + "\n")

if __name__ == "__main__":
	w2v = word2vec.Word2Vec.load(sys.argv[1])
	words = set()
	prefix = "../data/"
	for file_name in ["professions", "nationalities", "persons"]:
		terms = read_one_column(prefix + file_name)
		terms = map(normalize, terms)	
		S = all_words(terms)
		words = words.union(S)
	reduce_w2v(w2v, words, "../data/word2vec_reduced-2.txt")
