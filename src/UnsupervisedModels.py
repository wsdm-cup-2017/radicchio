import numpy as np
from _UnsupervisedBase import UnsupervisedBase
from scipy.sparse import lil_matrix 
from scipy import spatial 
from sklearn.decomposition import NMF
from utils import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier as RFC
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
    
    def train(self, path = "../data/profession.kb"):
        """
        Use NMF for training.
        Note that this is very memory-comsuming when our matrix is quite large.
        """
        M = lil_matrix((len(self.N_map), len(self.V_map)))
        with open(path, "r") as f:
            for line in f:
                (name, value) = line.strip().split("\t")
                M[self.N_map[name], self.V_map[value]] = 1
        self.M = M
        model = NMF(n_components = 15, max_iter = 1000)
        self.W = model.fit_transform(self.M)
        self.H = model.components_.T
    
class RandomGuess(UnsupervisedBase):
    """
    Randomly predict an ineger from 0 to 7.
    """
    def predict(self, pairs):
        return np.array([np.random.randint(8) for i in range(len(pairs))])
    def train(self):
        pass

class MeanGuess(UnsupervisedBase):
    """
    Randomly predict 3.5 .
    """
    def predict(self, pairs):
        return np.array([3.5 for i in range(len(pairs))])
    def train(self):
        pass
    
class KnowledgeBaseWV(UnsupervisedBase):
    def __init__(self, knowledge_base_path, w2v_path = "../models/vectors.bin"):
        self.w2v = word2vec.Word2Vec.load_word2vec_format(w2v_path, binary = True)
	self.w2v_dim = self.w2v[self.w2v.vocab.keys()[0]].shape[0]
	UnsupervisedBase.__init__(self)
        self.knowledge_base_path = knowledge_base_path
        self.learner = RFC(n_estimators = 1000, max_depth = 15, n_jobs = -1)
        #self.learner = SVC(kernel = "rbf", C = 10, probability = True) 
        #self.learner = LR(C = 10, max_iter = 1000) 

    def extract_features(self, pairs):
	X = []
        for pair in pairs:
	    vec1 = self.map_w2v(pair[0])
	    vec2 = self.map_w2v(pair[1])
	    x = np.hstack((vec1, vec2))
	    X.append(x)
	return  np.array(X)
        
    def train(self):
        self.scaler = StandardScaler()
	pairs, Y = read_labeled_data(labeled_data_path = self.knowledge_base_path)
	X = self.extract_features(pairs)
        X = self.scaler.fit_transform(X)
	self.learner.fit(X, Y)
    
    def predict(self, pairs):
        X = self.extract_features(pairs)
        X = self.scaler.transform(X)
        Y = self.learner.predict_proba(X)[:, 1]
	return self.scale_prediction(Y, pairs)
        
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

class KnowledgeBaseLearnersWV(KnowledgeBaseWV):
    def __init__(self, knowledge_base_path, all_values_path, w2v_path = "../models/vectors.bin"):
        values = read_one_column(all_values_path)
        UnsupervisedBase.__init__(self)
