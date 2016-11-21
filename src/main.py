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
	model = WordVector()
	model.evaluate(verbose = True)
