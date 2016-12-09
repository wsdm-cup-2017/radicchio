"""
Clustering Professions based on their Word2Vec embedding similarity.
"""

from utils import *
from gensim.models import word2vec
from scipy.spatial.distance import cosine

def generate_clusters(item_source, w2v, threshold):
	items = read_one_column(item_source)
	w2v_dim = w2v[w2v.vocab.keys()[0]].shape[0]

	cluster = {}

	item_vec_map = dict([(x, map_w2v(w2v, w2v_dim, x)) for x in items])

	index = 0
	cluster_id = 0
	for current in items:
		if index == 0:
			cluster[current] = "cluster" + str(cluster_id)
		else:
			best_link = max(items[:index], key=lambda prev: cosine(item_vec_map[prev], item_vec_map[current]))

			if cosine(item_vec_map[best_link], item_vec_map[current]) > threshold:
				cluster[current] = cluster[best_link]
			else:
				cluster_id += 1
				cluster[current] = "cluster" + str(cluster_id)
		index += 1
		
	print "Number of Clusters = ", cluster_id
	return cluster


def map_w2v(w2v, w2v_dim, text):
	term = normalize(text)
	word = "_".join(term)
	if word in w2v.vocab:
		return w2v[word]
	else:
		vec = np.zeros(w2v_dim)
		cnt = 0.0
		for word in term:
			if word in w2v.vocab:
				vec += w2v[word]
				cnt += 1
		if cnt > 0:
			vec /= cnt
		return vec

if __name__ == "__main__":
	w2v_path = "../models/word2vec_reduced.txt"
	professions_path = "../data/professions"

	w2v = word2vec.Word2Vec.load_word2vec_format(w2v_path)
	clusters = generate_clusters(professions_path, w2v, 0.9)
	print clusters
