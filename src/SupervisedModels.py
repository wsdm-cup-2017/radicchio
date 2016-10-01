import numpy as np
from _SupervisedBase import SupervisedBase

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
