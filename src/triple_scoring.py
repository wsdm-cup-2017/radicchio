import argparse
import glob
import os
from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
import numpy as np

"""
This script is our submission software.
"""
if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triple_scoring")
    parser.add_argument("-c", "--inputDataset", default = "../data/")
    parser.add_argument("-r", "--inputRun",  default = "../data/")
    parser.add_argument("-o", "--outputDir", default = "../output/") 
    args = parser.parse_args()
    np.random.seed(0)
    model = RandomGuess()
    if not os.path.exists(args.outputDir):
	os.mkdir(args.outputDir)
    for input_path in glob.glob(args.inputDataset + "/*.train"): 
	model.test(input_path, args.outputDir+ os.path.split(input_path)[1])
