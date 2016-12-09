import numpy as np
from abc import ABCMeta, abstractmethod
from utils import *
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
import cPickle
"""
This is the base abstract class for supervised models.
You have to implement the extract_features() method at least.
By default, it takes the support vector regression as the learning model (you are encouraged to modify it).
You should formulate your extracted features as a 2-D Numpy array (X) and  your labeled score as 1-D Numpy array  (Y).
For X, each row may epresents a feature vector of a (person-value) pair instance .

Todo :
	1. Model saving/loading
	2. Input the testing data and output the prediction in a file 
"""
class SupervisedBase(object):
	"""
	An abstract model class
	"""
	__metaclass__ = ABCMeta

	def __init__(self, learner_type = "SVR", parameters = {"kernel":"rbf", "C":10}):
		"""
		Use SVC with linear kernel and C = 0.1 as default
		You can add more different types of learners
		"""
		self.learner_type = learner_type
		if learner_type == "SVR":
			self.learner = SVR(kernel = parameters["kernel"], C = parameters["C"])
		elif learner_type == "7_SVM":
			self.learner = [SVC(kernel = parameters["kernel"], C = parameters["C"] ) for i in range(7)]

	def predict(self, X):
		"""
		Return : a 1-D Numpy array in which each element is the predicted score for each input pair (represented in feature vector)
		"""
                X = self.scaler.transform(X)
		if self.learner_type == "SVR":
			return self.normalize_prediction( self.learner.predict(X))
		elif self.learner_type == "7_SVM":
			Y = np.array([self.learner[i].predict(X) for i in range(7)])
			#return the sum of the predictions from 7 SVMs
			return self.normalize_prediction( np.sum(Y, axis = 0))
		
	def normalize_prediction(self, Y):
		Y[Y > 7] = 7
		Y[Y < 0] = 0
		return np.array(map(lambda x : int(round(x)), Y))

	def train(self, X, Y):
		self.scaler = StandardScaler()
		X = self.scaler.fit_transform(X)
		if self.learner_type == "SVR":
			self.learner.fit(X, Y)
		elif self.learner_type == "7_SVM":
			#train 7 SVMs
			for i in range(7):
				copyY  = Y.copy()
				copyY[Y < i+1] = 0
				copyY[Y >= i+1] = 1
				self.learner[i].fit(X, copyY)

	@abstractmethod
	def extract_features(self, pairs):
		"""
		You should formulate your extracted features as a 2-D Numpy array X.
		For X, each row may epresents a feature vector of a (person-value) pair instance .
		Return: 2-D Numpy array (X)
		NOTE : Remember to scale the features if you use a distance-based learner like SVR
		NOTE : You may store your features in the disk if it taks much time to extract
		"""
		raise NotImplementedError
	
        def train_and_save(self,  labeled_data_path , save_path, X_path = None):
		pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
		X = self.extract_features(pairs, X_path = X_path)
		self.train(X, Y)
		cPickle.dump((self.learner, self.scaler), open(save_path, "w"))

	def load(self, load_path):
		(self.learner, self.scaler) = cPickle.load(open(load_path, "r"))


	def test(self, input_path, output_path):
		"""
		Read from the input path and ouput the prediction to the output path.
		"""
		pairs = []
		with open(input_path, "r") as in_f:
			for line in in_f:
				pairs.append(line.strip().split("\t")[0:2])
		X = self.extract_features(pairs)
		Y = self.predict(X)
		with open(output_path, "w") as out_f:
			 for i, (name, value) in enumerate(pairs):
			 	out_f.write("%s\t%s\t%d\n" %(name, value, int(round(Y[i]))))
	
	def evaluate(self, labeled_data_path, X_path = None,  verbose = False, n_fold = 5): 
		"""
		For the supervised models, we measure the performance by 5-fold cross validation.
		"""
		
		#read labeled data
		pairs, Y = read_labeled_data(labeled_data_path = labeled_data_path)
		#extract features
		X = self.extract_features(pairs, X_path = X_path)
		#group person names
		Xs = []
		Ys = []
		Ps = []
		prev_group = None
		for i, p in enumerate(pairs):
			if p[0] != prev_group:
				prev_group = p[0]
				Xs.append([])
				Ys.append([])
				Ps.append([])
			Xs[-1].append(X[i].tolist())
			Ys[-1].append(Y[i])
			Ps[-1].append(p)
		for i in range(len(Xs)):
			Xs[i] = np.array(Xs[i])
			Ys[i] = np.array(Ys[i])
		
		#shuffle
		Xs, Ys, Ps = shuffleXY(Xs, Ys, Ps)
		

		fold_size = len(Xs) / n_fold
		in_acc, in_asd, in_tau = [], [], []
		val_acc, val_asd, val_tau = [], [], []
		for i in range(n_fold):
			trainXs, trainYs, testXs, testYs, trainPs, testPs = train_test_split(Xs, Ys, Ps, fold_size*i, fold_size*(i+1))	
			trainX = np.concatenate(trainXs)
			trainY = np.concatenate(trainYs)
			self.train(trainX, trainY)
			
			score1 = []
			score2 = []
			for testX, testY, testP in zip(testXs, testYs, testPs):
				predY = self.predict(testX)
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
			for trainX, trainY in zip(trainXs, trainYs):
				predY = self.predict(trainX)
				score1.append(predY.tolist())
				score2.append(trainY.tolist())
			in_acc.append(compute_acc(score1, score2))
			in_asd.append(compute_asd(score1, score2))
			in_tau.append(compute_tau(score1, score2))
		
		print "(In sample) Accuracy : %f, Distance : %f, Kendall tau : %f" % (np.mean(in_acc), np.mean(in_asd), np.mean(in_tau)) 
		print "(Validation) Accuracy : %f, Distance : %f, Kendall tau : %f" % (np.mean(val_acc), np.mean(val_asd), np.mean(val_tau)) 
