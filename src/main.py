from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
from FreebaseFeatures import FreebaseFeatures
from Ensemble import Ensemble
import numpy as np
import random

from utils import *

"""
This script evaluates the performance of several very basic models.
"""
if  __name__ == "__main__":

	random.seed(0)
	np.random.seed(0)
	
        professions = read_one_column("../data/professions")
	nationalities = read_one_column("../data/nationalities")
        profession_train = "../data/profession.train"
        nationality_train = "../data/nationality.train"

        print "======== RandomGuess ======="
	model = RandomGuess()
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
        print ""
    	
        print "======== MeanGuess ======="
	model = MeanGuess()
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
        print ""

	print "======== AllZeroFeature ======="
	model = AllZeroFeature()
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
        print ""
	
	print "======== Freebase IPCA ======="
	model = FreebaseFeatures(freebase_features="../data/freebase_features/features_ipca.bin", labels=professions)
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.train_and_save(profession_train, "../models/freebase_profession.mod")	

	model = FreebaseFeatures(freebase_features="../data/freebase_features/features_ipca.bin", labels=nationalities)
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
	model.train_and_save(nationality_train, "../models/freebase_nationality.mod")	
        print ""

        print "======== WordVector ======="
	model = WordVector(w2v_path = "../models/vectors.bin") 
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.train_and_save(profession_train, "../models/word2vec_profession.mod")	
        
	model = WordVector(w2v_path = "../models/vectors.bin") 
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
	model.train_and_save(nationality_train, "../models/freebase_nationality.mod")	
        print ""
	
        
	print "======== Ensemble ======="
	modelWP = WordVector(w2v_path = "../models/vectors.bin") 
	modelFP = FreebaseFeatures(freebase_features="../data/freebase_features/features_ipca.bin", labels=professions)
        model = Ensemble(model_list = [modelWP, modelFP])
	model.evaluate(labeled_data_path = profession_train, verbose=True)
	model.train_and_save(profession_train, "../models/ensemble_profession.mod")	

	modelWN = WordVector(w2v_path = "../models/vectors.bin") 
        modelFN = FreebaseFeatures(freebase_features="../data/freebase_features/features_ipca.bin", labels=nationalities)
        model = Ensemble(model_list = [modelWN, modelFN])
	model.evaluate(labeled_data_path = nationality_train, verbose=True)
	model.train_and_save(nationality_train, "../models/freebase_nationality.mod")	
        print ""
