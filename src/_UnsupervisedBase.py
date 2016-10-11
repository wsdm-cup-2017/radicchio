import numpy as np
from abc import ABCMeta, abstractmethod
from utils import read_labeled_data, get_distance, get_accuracy

"""
This is the base abstract class for unsupervised models.
You have to implement the extract_features(), train method(), and predict() method.
You may want to store your features into the obejct itself so that you can access it when training.
You may want to store your learning models into the obejct itself so that you can access it when predicting.
This UnsupervisedBase evaluates the performance on all the labeled scores.

Todo :
    1. Model saving/loading
    2. Input the testing data and output the prediction in a file 
"""

class UnsupervisedBase(object):
    """
    An abstract model class
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, pairs):
        """
        Input : a list of (person, value) pair
        Return : a 1-D Numpy array in which each element is the predicted score for each input pair
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def extract_features(self):
        """
        1. Read the data 
        2. Extract features
        3. Store the features in this object 
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
	
	Y = self.predict(pairs)
	with open(output_path, "w") as out_f:
	     for i, (name, value) in enumerate(pairs):
		 out_f.write("%s\t%s\t%d\n" %(name, value, int(round(Y[i]))))
    
    def evaluate(self, labeled_data_path = "../data/profession.train", verbose = False): 
        
        #read the data and extract features
        self.extract_features()
        
        #training
        self.train()
        
        #predicting
        pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
        predY = self.predict(pairs)
	if verbose is True:
            for i, (y, py) in enumerate(zip(Y, predY)):
	        print ", ".join(pairs[i]).ljust(50) , "True:", y, "/ Predicted:", py
        
        #evaluating
        distance = get_distance(Y, predY)
        accuracy = get_accuracy(Y, predY)
        print "Average Score Difference = %f, Accuracy = %f" %(distance, accuracy)
