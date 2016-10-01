import numpy as np
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
