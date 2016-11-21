import numpy as np
import re
import itertools
import sys
from random import shuffle
"""
This file contains some general functions to be used for other programs.
"""

def build_all_values_int(all_values_path = "../data/professions"):
	"""
	Read the values from the file and build a (value -> integer) dictionary mapping.
	"""
	cnt = 0
	V_map = {}
	with open(all_values_path, "r") as f:
		for line in f:
			value = line.strip()
			V_map[value] = cnt
			cnt +=1 
	return V_map
	
def build_all_names_int(all_names_path = "../data/persons"):
	"""
	Read the values from the file and build a (person name -> integer) dictionary mapping.
	"""
	cnt = 0
	N_map = {}
	with open(all_names_path, "r") as f:
		for line in f:
			name = line.strip().split("\t")[0]
			N_map[name] = cnt
			cnt +=1 
	return N_map

def read_labeled_data(labeled_data_path):
	"""
	Read the labeld data (.train) from the path.
	Return: a list of tuple (name, value) and its labeled scores. 
	"""
	truths = []
	pairs = []
	with open(labeled_data_path, "r") as f:
		for line in f:
			name, value, true_score = line.strip().split("\t")
			pairs.append((name, value))
			truths.append(float(true_score))
	return pairs, np.array(truths)

def normalize_name(name):
	"""
	Normalize a person name so that it can be mapped to the text
	"""
	name = "_".join(name.split()) #replace space with "_"
	name  = re.sub(r'[^\w\s]' , "", name.decode("utf-8"), re.UNICODE) #remove punctuations
	name = name.lower() # turn into lower case
	return name

def normalize_profession(profession):
	"""
	Normalize a profession so that it can be mapped to the text
	NOTE: Currently, we only return the last token 
	"""
	profession  = re.sub(r'[^\w\s]' , " ", profession.decode("utf-8"), re.UNICODE)#remove punctuations
	profession = profession.split()[-1] # only return the last token
	profession = profession.lower()# turn into lower case
	return profession


def get_distance(truths, preds):
	"""
	Calculate the mean of distances between the truths and the predictions.
	"""
	return np.mean([abs(y-py) for y, py in zip(truths, preds)])

def get_accuracy(truths, preds):
	"""
	Calculate the mean of accuracies between the truths and the predictions.
	"""
	return np.mean([1.0 if abs(y-py) <= 2 else 0.0 for y, py in zip(truths, preds)])

def kendall_tau_ranks(scores):
	# Compute buckets of the same score.
	buckets = {}
	for i, s in enumerate(scores):
		if s not in buckets:
			buckets[s] = []
		buckets[s].append(i)
	# Iterate over buckets and distribute ranks.
	last_rank = 0
	ranks = list(range(0, len(scores)))
	for s in sorted(buckets.keys()):
		n = len(buckets[s])
		# Average ties
		rank = last_rank + ((n + 1) / float(2))
		for i in buckets[s]:
			ranks[i] = rank
		last_rank += n
	return ranks


def kendall_tau(scores1, scores2, p = 0.5):
	if len(scores1) == 1:
		return 0.0
	# The ranks of the scores. Equal 
	ranks1 = kendall_tau_ranks(scores1)
	ranks2 = kendall_tau_ranks(scores2)
	# All possible pairs i, j with i < j.
	pairs = itertools.combinations(range(0, len(scores1)), 2)
	penalty = 0.0
	# Count the number of pairs in the second list (the ground truth), where
	# pairs with equal scores count only p (default 0.5, see above).
	num_ordered = 0.0
	for i, j in pairs:
		# The ranks of scores i and j in both score lists.
		a_i = ranks1[i]
		a_j = ranks1[j]
		b_i = ranks2[i]
		b_j = ranks2[j]
		# CASE 1: Scores i and j are different in both lists. Then there is a
		# penalty of 1.0 iff the order does not match.
		if a_i != a_j and b_i != b_j:
			if (a_i < a_j and b_i < b_j) or (a_i > a_j and b_i > b_j):
				pass
			else:
				penalty += 1
		# CASE 2: Scores i and j are the same in both lists. Then there is no
		# penalty.
		elif a_i == a_j and b_i == b_j:
			pass
		# CASE 3: Scores i and j are the same in one list, but different in the
		# other. Then there is a penalty of p (default value 0.5, see above).
		else:
			penalty += p
		# Count this pair as 1 if the scores in list 2 are different, otherwise
		# as p (default value 0.5, see above).
		if b_i != b_j:
			num_ordered += 1
		else:
			num_ordered += p
	# Return the average penalty.
	return penalty / num_ordered


def compute_acc(scores1, scores2, delta = 2):
	num_all = 0.0
	num_acc = 0.0
	for group1, group2 in zip(scores1, scores2):
		for score1, score2 in zip(group1, group2):
			num_all += 1
			if abs(score1 - score2) <= delta:
				num_acc += 1
	return num_acc / num_all

def compute_asd(scores1, scores2):
	num_all = 0.0
	sum_difference = 0.0
	for group1, group2 in zip(scores1, scores2):
		for score1, score2 in zip(group1, group2):
			num_all += 1
			sum_difference += abs(score1 - score2)
	return sum_difference / num_all


def compute_tau(scores1, scores2):
	num_groups = len(scores1)
	sum_tau = 0.0
	for group1, group2 in zip(scores1, scores2):
		sum_tau += kendall_tau(group1, group2)
	return sum_tau / num_groups

def shuffleXY(X, Y, pairs):
	X_shuffle = []
	Y_shuffle = []
	P_shuffle = []
	index_shuffle = range(len(X))
	shuffle(index_shuffle)
	for i in index_shuffle:
		X_shuffle.append(X[i])
		Y_shuffle.append(Y[i])
		P_shuffle.append(pairs[i])
	return X_shuffle, Y_shuffle, P_shuffle

def train_test_split(X, Y, P, base, end):
	trainX = X[:base] + X[end:]
	trainY = Y[:base] + Y[end:]
	trainP = P[:base] + P[end:]
	testX = X[base:end]
	testY = Y[base:end]
	testP = P[base:end]
	return trainX, trainY, testX, testY, trainP, testP
