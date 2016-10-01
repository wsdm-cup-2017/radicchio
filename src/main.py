from UnsupervisedModels import AllMF, RandomGuess, MeanGuess
from SupervisedModels import AllZeroFeature
import numpy as np

"""
This script evaluates the performance of several very basic models.
"""
if  __name__ == "__main__":
    
    np.random.seed(0)
    
    print "======== RandomGuess ======="
    model = RandomGuess()
    model.evaluate()
    
    print "======== MeanGuess ======="
    model = MeanGuess()
    model.evaluate()

    print "======== AllMF ======="
    model = AllMF()
    model.evaluate()
    
    print "======== AllZeroFeature ======="
    model = AllZeroFeature()
    model.evaluate()
