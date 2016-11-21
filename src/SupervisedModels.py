import numpy as np
from _SupervisedBase import SupervisedBase
from sklearn.preprocessing import StandardScaler
from utils import *
from gensim.models import word2vec

"""
This file implements several supervised learning models that inherit from the SupervisedBase.
"""
class AllZeroFeature(SupervisedBase):
	"""
	Generate 2-dimensional features for each instance. 
	All the features are 0.
	"""
	def extract_features(self, pairs):
		return np.zeros((len(pairs),2))

class WordVector(SupervisedBase):
	"""
	Use word vectors as features.
	Currently, we only support loading fetaures from files.
	"""
	def __init__(self, w2v_path = "../models/word2vec_reduced.txt"):
		if w2v_path is not None:
			self.w2v = word2vec.Word2Vec.load_word2vec_format(w2v_path)
			self.w2v_dim = self.w2v[self.w2v.vocab.keys()[0]].shape[0]
		else:
			self.w2v = None
		SupervisedBase.__init__(self)

	def extract_features(self, pairs, X_path = None):
		if self.w2v is not None or X_path is None:
			X = []
			for pair in pairs:
				vec1 = self.map_w2v(pair[0])
				vec2 = self.map_w2v(pair[1])
				x = np.hstack((vec1, vec2))
				X.append(x)
			X = np.array(X)
			#np.save("../data/X_profession.npy", X)
			#np.save("../data/X_nationality.npy", X)
		else:
			X = np.load(X_path)

		#standardly scale features
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
		return X

	def map_w2v(self, text):
		term = normalize(text)
		word = "_".join(term)
		if word in self.w2v.vocab:
			return self.w2v[word]
		else:
			vec = np.zeros(self.w2v_dim)
			cnt = 0.0
			for word in text:
				if word in self.w2v.vocab:
					vec += self.w2v[word]
					cnt += 1
			if cnt >= 0:
				vec /= cnt
			return vec
