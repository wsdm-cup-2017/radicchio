import numpy as np

from _SupervisedBase import SupervisedBase

class CombinedModel(SupervisedBase):
        def __init__(self, model_list):
                SupervisedBase.__init__(self)
                self.models = model_list

        def extract_features(self, pairs, X_path):
                return np.concatenate([m.extract_features(pairs, X_path) for m in self.models], axis=1)

