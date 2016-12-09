import cPickle
import numpy as np
from utils import *

class Ensemble(object):
    def __init__(self, model_list):
        self.model_list = model_list       
    
    def train(self, pairs, Y):
        for model in self.model_list:
            X = model.extract_features(pairs)
            model.train(X, Y)

    def predict(self, pairs):
        Ys = []
        for model in self.model_list:
            X = model.extract_features(pairs)
            Y = model.predict(X)
            Ys.append(Y)
        Y = np.array(Ys).mean(axis = 0)
        return self.normalize_prediction(Y)

    def normalize_prediction(self, Y):
	Y[Y > 7] = 7
	Y[Y < 0] = 0
	return np.array(map(lambda x : int(round(x)), Y))
    
    def train_and_save(self,  labeled_data_path , save_path):
        for i, model in enumerate(self.model_list):
            model.train_and_save(labeled_data_path, save_path+str(i))

    def load(self, load_path):
        for i, model in enumerate(self.model_list):
            model.load(load_path+ str(i))
    
    def test(self, input_path, output_path):
        pairs = []
        with open(input_path, "r") as in_f:
            for line in in_f:
                pairs.append(line.strip().split("\t")[0:2])
                Y = self.predict(pairs)
        with open(output_path, "w") as out_f:
            for i, (name, value) in enumerate(pairs):
                out_f.write("%s\t%s\t%d\n" %(name, value, int(round(Y[i]))))
            
    def evaluate(self, labeled_data_path, verbose = False, n_fold = 5): 
	
	#read labeled data
	pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
	#group person names
	Ys = []
	Ps = []
	prev_group = None
	for i, p in enumerate(pairs):
		if p[0] != prev_group:
			prev_group = p[0]
			Ys.append([])
			Ps.append([])
		Ys[-1].append(Y[i])
		Ps[-1].append(p)
	for i in range(len(Ys)):
		Ys[i] = np.array(Ys[i])
	
	#shuffle
	Ys, Ps = shufflePY(Ys, Ps)
	

	fold_size = len(Ys) / n_fold
	in_acc, in_asd, in_tau = [], [], []
	val_acc, val_asd, val_tau = [], [], []
	for i in range(n_fold):
		trainYs, testYs, trainPs, testPs = train_test_split_PY(Ys, Ps, fold_size*i, fold_size*(i+1))	
		trainY = np.concatenate(trainYs)
                trainP = [] 
                for pa in trainPs:
                    for p in pa:
                        trainP.append(p)
                self.train(trainP, trainY)
		
		score1 = []
		score2 = []
		for testY, testP in zip(testYs, testPs):
			predY = self.predict(testP)
			score1.append(predY.tolist())
			score2.append(testY.tolist())
			if verbose is True:
				for py, y, p in zip(predY, testY, testP):
					print ", ".join(p).ljust(50) , "True:", y, "/ Predicted:", py
		val_acc.append(compute_acc(score1, score2))
		val_asd.append(compute_asd(score1, score2))
		val_tau.append(compute_tau(score1, score2))
		
		score1 = []
		score2 = []
		for trainP, trainY in zip(trainPs, trainYs):
			predY = self.predict(trainP)
			score1.append(predY.tolist())
			score2.append(trainY.tolist())
		in_acc.append(compute_acc(score1, score2))
		in_asd.append(compute_asd(score1, score2))
		in_tau.append(compute_tau(score1, score2))
	
	print "(In sample) Accuracy : %f, Distance : %f, Kendall tau : %f" % (np.mean(in_acc), np.mean(in_asd), np.mean(in_tau)) 
	print "(Validation) Accuracy : %f, Distance : %f, Kendall tau : %f" % (np.mean(val_acc), np.mean(val_asd), np.mean(val_tau)) 
