import numpy as np
from _UnsupervisedBase import UnsupervisedBase
from scipy.sparse import lil_matrix 
from scipy import spatial 
from sklearn.decomposition import NMF
from utils import *
from sklearn.svm import SVR
from gensim.models import word2vec
from sklearn.preprocessing import StandardScaler

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
class ZeroSeven(UnsupervisedBase):
    def __init__(self, label_path):
        self.labeled_data_path = label_path
        pairs, Y = read_labeled_data(labeled_data_path = self.labeled_data_path)
    def predict(self, pairs):
    def train(self):
        
    def extract_features(self):
        pass
    def normalize_prediction(self, Y):
	Y[Y > 7] = 7
	Y[Y < 0] = 0
	return np.array(map(lambda x : int(round(x)), Y))
    
class ZeroSevenWV(UnsupervisedBase):
    def __init__(self, label_path, w2v_path = "../models/vectors.bin"):
	self.w2v = word2vec.Word2Vec.load_word2vec_format(w2v_path, binary = True)
	self.w2v_dim = self.w2v[self.w2v.vocab.keys()[0]].shape[0]
	UnsupervisedBase.__init__(self)
        self.labeled_data_path = label_path
	self.learner = SVR(kernel = "rbf", C = 1)

    def extract_features(self, pairs = None):
        if pairs is None:
	    pairs, Y = read_labeled_data(labeled_data_path = self.labeled_data_path)
	X = []
        for pair in pairs:
	    vec1 = self.map_w2v(pair[0])
	    vec2 = self.map_w2v(pair[1])
	    x = np.hstack((vec1, vec2))
	    X.append(x)
	X = np.array(X)
        self.X = X

    def train(self):
        self.scaler = StandardScaler()
	pairs, Y = read_labeled_data(labeled_data_path = self.labeled_data_path)
	X = self.scaler.fit_transform(self.X)
	self.learner.fit(X, Y)
    
    def normalize_prediction(self, Y):
	Y[Y > 7] = 7
	Y[Y < 0] = 0
	return np.array(map(lambda x : int(round(x)), Y))
    
    def predict(self, pairs):
	pairs, Y = read_labeled_data(labeled_data_path = self.labeled_data_path)
        self.extract_features(pairs)
        X = self.scaler.transform(self.X)
	return self.normalize_prediction( self.learner.predict(X))
        

    def map_w2v(self, text):
	term = normalize(text)
	word = "_".join(term)
	if word in self.w2v.vocab:
	    return self.w2v[word]
	else:
	    vec = np.zeros(self.w2v_dim)
	    cnt = 0.0
	    for word in term:
		if word in self.w2v.vocab:
		    vec += self.w2v[word]
		    cnt += 1
	    if cnt > 0:
		vec /= cnt
	    return vec

    def train_and_save(self):
	X = self.extract_features(pairs, X_path = X_path)
	self.train(X, Y)
	tmp = self.w2v
	self.w2v = None
	cPickle.dump((self.learner, self.scaler), open(save_path, "w"))
	self.w2v = tmp
    def load(self, load_path):
	(self.learner, self.scaler) = cPickle.load(open(load_path, "r"))

