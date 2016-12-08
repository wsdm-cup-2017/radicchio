"""
Classifier that extracts Freebase features and uses them for multiclass classification.
"""

import shelve

import numpy as np
import scipy.sparse as sps

from _SupervisedBase import SupervisedBase

class FreebaseFeatures(SupervisedBase):

	def __init__(self, freebase_features, labels):
		assert labels is not None and isinstance(labels, list)
		
		SupervisedBase.__init__(self)

		self.features = shelve.open(freebase_features, 'r')

		# Sample Freebase Id for Alfred Hitchcock
		sample = self.features['m.0j_c']
		self.isSparse = sps.issparse(sample)

		if self.isSparse:
				self.feature_size = sample.toarray().flatten().size
		else:
				self.feature_size = sample.size

		self.mapping = {}
		with open('../../triple-scoring/persons', 'r') as mapping_file:
			for line in mapping_file.xreadlines():
				parts = line.strip().split('\t')
				name = parts[0]
				self.mapping[name] = parts[1]

		self.labels = {}
		index = 0
		for lab in labels:
			self.labels[lab.strip()] = index
			index += 1

		self.num_labels = index

	def extract_features(self, pairs, X_path=None):
		X_feat = np.zeros((len(pairs), self.feature_size * self.num_labels))

		idx = 0
		for (person, label) in pairs:
			if person in self.mapping and self.mapping[person] in self.features:
				pos = self.labels[label]

				features_raw = self.features[self.mapping[person]]
				feature_val = np.zeros(self.feature_size * self.num_labels)

				if self.isSparse:
					features_raw = features_raw.toarray()

				bounds = (pos * self.feature_size, (pos + 1) * self.feature_size)
				feature_val[bounds[0]: bounds[1]] += features_raw.flatten()

				X_feat[idx, :] += feature_val
				idx += 1

		return X_feat
