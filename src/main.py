from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
from FreebaseFeatures import FreebaseFeatures
import numpy as np
import random

from utils import *

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

	professions = read_one_column("../data/professions")
	nationalities = read_one_column("../data/nationalities")
	
	print "======== Freebase IPCA ======="
	model_professions = FreebaseFeatures(freebase_features="../freebase_features/features_ipca.bin", labels=professions)
	model_professions.evaluate(labeled_data_path = "../data/profession.train", verbose=True)

	model_nationality = FreebaseFeatures(freebase_features="../freebase_features/features_ipca.bin", labels=nationalities)
	model_nationality.evaluate(labeled_data_path = "../data/nationality.train")

	"""
	print "======== WordVector ======="
	model = WordVector(w2v_path = None) 
	model.evaluate(labeled_data_path = "../data/profession.train", X_path = "../data/X_profession.npy", verbose = True)
	model.evaluate(labeled_data_path = "../data/nationality.train", X_path = "../data/X_nationality.npy", verbose = True)
	
	modelP = WordVector(w2v_path = None) 
	mod_path = "../models/profession.mod"
	input_path = "../data/profession.train"
	X_path = "../data/X_profession.npy"
	modelP.train_and_save(input_path, mod_path, X_path = X_path)	
	modelP.load(mod_path)
	
	modelN = WordVector(w2v_path = None) 
	mod_path = "../models/nationality.mod"
	input_path = "../data/nationality.train"
	X_path = "../data/X_nationality.npy"
	modelN.train_and_save(input_path, mod_path, X_path = X_path)	
	modelN.load(mod_path)
