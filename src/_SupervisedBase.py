import numpy as np
from abc import ABCMeta, abstractmethod
from utils import read_labeled_data, get_distance, get_accuracy
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
#new in sklearn 0.18
from sklearn.model_selection import GroupKFold, KFold 

"""
This is the base abstract class for supervised models.
You have to implement the extract_features() method at least.
By default, it takes the support vector regression as the learning model (you are encouraged to modify it).
You should formulate your extracted features as a 2-D Numpy array (X) and  your labeled score as 1-D Numpy array  (Y).
For X, each row may epresents a feature vector of a (person-value) pair instance .

Todo :
    1. Model saving/loading
    2. Input the testing data and output the prediction in a file 
"""
class SupervisedBase(object):
    """
    An abstract model class
    """
    __metaclass__ = ABCMeta

    def __init__(self, learner_type = "SVR", parameters = {"kernel":"rbf", "C":10}):
        """
        Use SVC with linear kernel and C = 0.1 as default
        You can add more different types of learners
	"""
	self.learner_type = learner_type
	if learner_type == "SVR":
	    self.learner = SVR(kernel = parameters["kernel"], C = parameters["C"])
	elif learner_type == "7_SVM":
	    self.learner = [SVC(kernel = parameters["kernel"], C = parameters["C"] ) for i in range(7) ]
	elif learner_type == "RandomForest":
		self.learner = RandomForestRegressor(n_estimators=parameters["n_estimators"], n_jobs=4)
        
    def predict(self, X):
        """
        Return : a 1-D Numpy array in which each element is the predicted score for each input pair (represented in feature vector)
        """
	if self.learner_type == "SVR" or self.learner_type == "RandomForest":
	    return np.array([ round(y) for y in self.learner.predict(X)])
        elif self.learner_type == "7_SVM":
	    Y = np.array([self.learner[i].predict(X) for i in range(7)])
    	    #return the sum of the predictions from 7 SVMs
	    return np.sum(Y, axis = 0)

    def train(self, X, Y):
	if self.learner_type == "SVR" or self.learner_type == "RandomForest":
	    self.learner.fit(X, Y)
        elif self.learner_type == "7_SVM":
	    #train 7 SVMs
	    for i in range(7):
            	copyY  = Y.copy()
	    	copyY[Y < i+1] = 0
	    	copyY[Y >= i+1] = 1
	    	self.learner[i].fit(X, copyY)

    @abstractmethod
    def extract_features(self, pairs):
        """
        You should formulate your extracted features as a 2-D Numpy array X.
        For X, each row may epresents a feature vector of a (person-value) pair instance .
        Return: 2-D Numpy array (X)
        NOTE : Remember to scale the features if you use a distance-based learner like SVR
        NOTE : You may store your features in the disk if it taks much time to extract
        """
        raise NotImplementedError
    
    def test(self, input_path, output_path):
	"""
	Read from the input path and ouput the prediction to the output path.
	"""
	pairs = []
	with open(input_path, "r") as in_f:
	    for line in in_f:
		pairs.append(line.strip().split("\t")[0:2])
	X = self.extract_features(pairs)
	Y = self.predict(X)
	with open(output_path, "w") as out_f:
	     for i, (name, value) in enumerate(out_f):
		 out_f.write("%s\t%s\t%d\n" %(name, value, int(round(Y[i]))))

    def evaluate(self, labeled_data_path = "../data/profession.train", verbose = False): 
        """
        For the supervised models, we measure the performance by 5-fold cross validation.
        """
        #kf = KFold(n_splits = 5, shuffle = True, random_state = 0) 
        kf = GroupKFold(n_splits = 5) 
        
        #read labeled data
        pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
        
	#group person names
	distinct_name_list = list(set([name for name, value in pairs]))
	groups = np.array([distinct_name_list.index(name) for name, value in pairs])

        #extract features
        X = self.extract_features(pairs)
        
        #cross validation
        in_distances = []
        in_accuracies = []
        val_distances = []
        val_accuracies = []
        for train_index, valid_index in kf.split(X, Y, groups):
            trainX, validX = X[train_index], X[valid_index]     
            trainY, validY = Y[train_index], Y[valid_index]     
            self.train(trainX, trainY)
            
	    #evalute : in sample
	    predY = self.predict(trainX)
            in_distances.append(get_distance(trainY, predY))
            in_accuracies.append(get_accuracy(trainY, predY))
	    
	    #evaluate : validation
	    predY = self.predict(validX)
	    val_distances.append(get_distance(validY, predY))
            val_accuracies.append(get_accuracy(validY, predY))
	    if verbose is True:
                val_pairs = np.array(pairs)[valid_index]
		for i, (y, py) in enumerate(zip(validY, predY)):
		    print ", ".join(val_pairs[i]).ljust(50) , "True:", y, "/ Predicted:", py
            
        #print out the evaluation
        in_distance = np.mean(in_distances)
        in_accuracy = np.mean(in_accuracies)
        val_distance = np.mean(val_distances)
        val_accuracy = np.mean(val_accuracies)
        print "(In Sample)Average Score Difference = %f, Accuracy = %f" %(in_distance, in_accuracy)
        print "(Validated)Average Score Difference = %f, Accuracy = %f" %(val_distance, val_accuracy)
