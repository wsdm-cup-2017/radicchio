import numpy as np
from abc import ABCMeta, abstractmethod
from utils import read_labeled_data, get_distance, get_accuracy
from sklearn.svm import SVR

#new in sklearn 0.18
from sklearn.model_selection import KFold 

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

    def __init__(self):
        """
        Use SVR with linear kernel and C = 0.1 as default
        You may overwrite it and tune the parameters by yourself
        """
        self.learner = SVR(kernel = "linear", C = 0.1)

    def predict(self, X):
        """
        Return : a 1-D Numpy array in which each element is the predicted score for each input pair (represented in feature vector)
        """
        return self.learner.predict(X)

    def train(self, X, Y):
        self.learner.fit(X, Y)

    @abstractmethod
    def extract_features(self):
        """
        You should formulate your extracted features as a 2-D Numpy array X.
        For X, each row may epresents a feature vector of a (person-value) pair instance .
        Return: 2-D Numpy array (X)
        NOTE : Remember to scale the features if you use a distance-based learner like SVR
        NOTE : You may store your features in the disk if it taks much time to extract
        """
        raise NotImplementedError
    
    def evaluate(self, labeled_data_path = "../data/profession.train"): 
        """
        For the supervised models, we measure the performance by 5-fold cross validation.
        """
        kf = KFold(n_splits = 5, shuffle = True, random_state = 0) 
        
        #read labeled data
        pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
        
        #extract features
        X = self.extract_features(pairs)
        
        #cross validation
        distances = []
        accuracies = []
        for train_index, valid_index in kf.split(X):
            trainX, validX = X[train_index], X[valid_index]     
            trainY, validY = Y[train_index], Y[valid_index]     
            self.train(trainX, trainY)
            predY = self.predict(validX)
            distances.append(get_distance(validY, predY))
            accuracies.append(get_accuracy(validY, predY))
        
        #print out the evaluation
        distance = np.mean(distances)
        accuracy = np.mean(accuracies)
        print "Average Score Difference = %f, Accuracy = %f" %(distance, accuracy)
