import numpy as np
from _SupervisedBase import SupervisedBase
from sklearn.preprocessing import StandardScaler
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
    def extract_features(self, pairs, load_from_X = True, X_path = "../data/X_name_profession_600.np"):
	scaler = StandardScaler()
	if load_from_X is True:
	    X = np.load(X_path)
	
	#standardly scale features
	X = scaler.fit_transform(X)
        return X
