from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
import numpy as np
import random

"""
This script evaluates the performance of several very basic models.
"""
if  __name__ == "__main__":

	random.seed(0)
	np.random.seed(0)
	print "======== RandomGuess ======="
	model = RandomGuess()
	model.evaluate(verbose = False)
	""" 
	print "======== MeanGuess ======="
	model = MeanGuess()
	model.evaluate()

	print "======== AllMF ======="
	model = AllMF()
	model.evaluate()
	
	print "======== AllZeroFeature ======="
	model = AllZeroFeature()
	model.evaluate()
	""" 
	print "======== WordVector ======="
	#model = WordVector()
	#model.evaluate(labeled_data_path = "../data/nationality.train", verbose = True)
	
	model = WordVector(w2v_path = None) 
	model.evaluate(labeled_data_path = "../data/profession.train", X_path = "../data/X_profession.npy", verbose = True)
	model.evaluate(labeled_data_path = "../data/nationality.train", X_path = "../data/X_nationality.npy", verbose = True)
