import numpy as np
from abc import ABCMeta, abstractmethod
from utils import *

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
        
        def scale_prediction(self, Y , pairs):
	    
            Ps = [[]]
            for y, pair in zip(Y, pairs):
                if len(Ps[-1]) == 0 or Ps[-1][0][1] == pair[0]:
                    Ps[-1].append((y, pair[0]))
                else:
                    Ps.append([(y, pair[0])])
            scaledY = []
            for group in Ps:
                max_y = max([y for y, p in group])
                for y, p in group:
                    scaledY.append(int(round(float(y) / max_y *7)))
            return np.array(scaledY)

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
		
		#training
		self.train()
		
		#predicting
		pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
		predY = self.predict(pairs)
		if verbose is True:
		    for py, y, p in zip(predY, Y, pairs):
			print ", ".join(p).ljust(50) , "True:", y, "/ Predicted:", py
		score1 = []
		score2 = []
		prev_group = None
		for i, pair in enumerate(pairs):
			g = pair[0]
			if prev_group != g:
				prev_group = g
				score1.append([])
				score2.append([])
			score1[-1].append(predY[i])
			score2[-1].append(Y[i])
		acc = compute_acc(score1, score2)
		asd = compute_asd(score1, score2)
		tau = compute_tau(score1, score2)
		print "Accuracy : %f, Distance : %f, Kendall tau : %f" % (np.mean(acc), np.mean(asd), np.mean(tau)) 
