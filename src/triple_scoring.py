#!/usr/bin/python
import argparse
import glob
import os
from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
import numpy as np
import cPickle
from utils import *


def check_type(file_path, professions, nationalities):
	hit_p = 0.0
	hit_n = 0.0
	with open(file_path, "r") as f:
		for line in f:
			value = line.strip().split("\t")[1]
			if value in professions:
				hit_p += 1
			if value in nationalities:
				hit_n += 1
	if hit_p >= hit_n:
		return "profession"
	else:
		return "nationality"
	
"""
This script is our submission software.
"""
if __name__ == "__main__":
	#parse arguments
	np.random.seed(0)
	parser = argparse.ArgumentParser(description="triple_scoring")
	parser.add_argument("-c", "--inputDataset", default = "../data/")
	parser.add_argument("-r", "--inputRun",  default = "../data/")
	parser.add_argument("-i", "--inputFiles", action = "append", nargs = "+")
	parser.add_argument("-o", "--outputDir", default = "../output/")
	parser.add_argument("-m", "--modelDir", default = "../models/")

	args = parser.parse_args()
	
	if not os.path.exists(args.outputDir):
		os.mkdir(args.outputDir)
	
	#load professions / nationality
	professions = read_one_column(os.path.join(args.inputDataset, "professions"))
	nationalities = read_one_column(os.path.join(args.inputDataset, "nationalities"))

	
	#load models
	w2v_path = os.path.join(args.modelDir, "word2vec_reduced.txt")
	modelP = WordVector(w2v_path = w2v_path) 
	modelP.load(os.path.join(args.modelDir, "profession.mod"))
	modelN = WordVector(w2v_path = None) 
	modelN.w2v = modelP.w2v
	modelN.w2v_dim = modelP.w2v_dim
	modelN.load(os.path.join(args.modelDir, "nationality.mod"))
	
        try:
            if args.inpusFiles is not None:
                for input_path in args.inputFiles:
                        input_path = input_path[0]
                        input_type = check_type(input_path, professions, nationalities)
                        file_name = os.path.split(input_path)[1]
                        if input_type == "profession":
                                modelP.test(input_path, os.path.join(args.outputDir, file_name))
                        elif input_type == "nationality":
                                modelN.test(input_path, os.path.join(args.outputDir, file_name))
        except:
            pass
	if os.path.isfile(args.inputDataset):
		file_name = os.path.split(args.inputDataset)[1]
                file_path = args.inputDataset
                input_type = check_type(file_path, professions, nationalities)
                if input_type == "profession":
                        modelP.test(file_path, os.path.join(args.outputDir, file_name))
                elif input_type == "nationality":
                        modelN.test(file_path, os.path.join(args.outputDir, file_name))
	else:
		for file_path in glob.glob(args.inputDataset + "/*.train"):
		        file_name = os.path.split(file_path)[1]
                        input_type = check_type(file_path, professions, nationalities)
                        if input_type == "profession":
                                modelP.test(file_path, os.path.join(args.outputDir, file_name))
                        elif input_type == "nationality":
                                modelN.test(file_path, os.path.join(args.outputDir, file_name))
		for file_path in glob.glob(args.inputDataset + "/*.test"):
		        file_name = os.path.split(file_path)[1]
                        input_type = check_type(file_path, professions, nationalities)
                        if input_type == "profession":
                                modelP.test(file_path, os.path.join(args.outputDir, file_name))
                        elif input_type == "nationality":
                                modelN.test(file_path, os.path.join(args.outputDir, file_name))
