import numpy as np
from _UnsupervisedBase import UnsupervisedBase
from scipy.sparse import lil_matrix 
from scipy import spatial 
from sklearn.decomposition import NMF
from utils import build_all_names_int, build_all_values_int

class AllMF(UnsupervisedBase):
    """
    This class implements the matrix factorization algorithm.
    The matrix we factorize is that one built by all person-value pairs (without labeled scores) in .kb files.
    Each row is a person name, each column is a kind of value.
    1 means this person-value pair is in our .kb file, and 0 means not.
    We store it in a Scipy sparse matrix
    """
    def __init__(self):
        #mapping from names to integers
        self.N_map = build_all_names_int()
        #mapping from values to integers
        self.V_map = build_all_values_int()

    def predict(self, pairs):
        """
            Use the reconstruction score (W[person] \dot H[value]) as the prediction
            W is the low-rank approximation of person names.
            H is the low-rank approximation of values.
        """
        def cosine_similarity(A, B):
            return 1.0 - spatial.distance.cosine(A, B)
        preds = []
        for pair in pairs:
            #cosine similarity : very poor now
            #pred = cosine_similarity(self.W[self.N_map[pair[0]]] ,self.H[self.V_map[pair[1]]])*7
            
            #use the reconstruction score (W[person] \dot H[value]) as the prediction
            pred = self.W[self.N_map[pair[0]]].dot(self.H[self.V_map[pair[1]]])*7.0 
            preds.append(pred)
        return np.array(preds)
    
    def train(self):
        """
        Use NMF for training.
        Note that this is very memory-comsuming when our matrix is quite large.
        """
        model = NMF(n_components = 15, max_iter = 1000)
        self.W = model.fit_transform(self.M)
        self.H = model.components_.T
    
    def extract_features(self, path = "../data/profession.kb"):
        """
        Build the person-value matrix.
        """
        M = lil_matrix((len(self.N_map), len(self.V_map)))
        with open(path, "r") as f:
            for line in f:
                (name, value) = line.strip().split("\t")
                M[self.N_map[name], self.V_map[value]] = 1
        self.M = M

class RandomGuess(UnsupervisedBase):
    """
    Randomly predict an ineger from 0 to 7.
    """
    def predict(self, pairs):
        return np.array([np.random.randint(8) for i in range(len(pairs))])
    def train(self):
        pass
    def extract_features(self):
        pass

class MeanGuess(UnsupervisedBase):
    """
    Randomly predict 3.5 .
    """
    def predict(self, pairs):
        return np.array([3.5 for i in range(len(pairs))])
    def train(self):
        pass
    def extract_features(self):
        pass


