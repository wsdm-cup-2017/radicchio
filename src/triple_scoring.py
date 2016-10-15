import argparse
import glob
import os
from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature, WordVector
import numpy as np

"""
This script is our submission software.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triple_scoring")
    parser.add_argument("-c", "--inputDataset", default = "../data/")
    parser.add_argument("-r", "--inputRun",  default = "../data/")
    parser.add_argument("-o", "--outputDir", default = "../output/")

    args = parser.parse_args()
    np.random.seed(0)
    model = RandomGuess()

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    if os.path.isfile(args.inputDataset):
        fileName = os.path.split(args.inputDataset)[1]
        model.test(args.inputDataset, os.path.join(args.outputDir, fileName))
    else:
        for input_path in glob.glob(args.inputDataset + "/*.train"):
            fileName = os.path.split(input_path)[1]
            model.test(input_path, os.path.join(args.outputDir, fileName))
