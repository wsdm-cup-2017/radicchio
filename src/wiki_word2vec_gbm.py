# -*- coding: utf-8 -*-
"""
@author: Jay
"""
"""
GBM Regression, Classification and LASSO regression using features from word embeddings ,online wiki scarping 
and manual clustering results


Wiki features scraped online from wiki_scrape.py did not work
manual clustering improved results slighlty on distance and accuracy.
"""
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import math
import os
import re
import matplotlib.pylab as plt
from gensim.models import word2vec

%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv( "C:/triple-scoring/profession.train", header=None, 
 delimiter="\t", quoting=3 )
train.columns=['Person','Profession','Score'] 


train_pr = pd.read_csv( "C:/triple-scoring/professions_clustered.csv", header=None, 
 delimiter=",", quoting=3 )
train_pr.columns=['Profession','cluster'] 
#train_pr.to_csv('cluster_profession.csv',sep=",")
for k in range(len(train_pr['cluster'])):
    train_pr['cluster'][k] = re.sub("[^a-zA-Z0-9]", "", train_pr['cluster'][k])
    train_pr['Profession'][k]=re.sub("[^a-zA-Z0-9]", "", train_pr['Profession'][k])
for p in range(len(train['Profession'])):
    train['Profession'][p]=re.sub("[^a-zA-Z0-9]", "", train['Profession'][p])


tr_clust=[]
for r in range(len(train['Profession'])):
    for k in range(len(train_pr['cluster'])) :
        if train['Profession'][r]==train_pr['Profession'][k]:
            tr_clust.append(train_pr['cluster'][k])
train['cluster']=tr_clust            
lines=[]
for i in range(train.shape[0]):
    lines.append(str(train['Person'][i])+str("\t")+str(train['Profession'][i]))
rec=[]
for k in range(len(lines)):
    rec.append(lines[k].split('\t'))  

model = word2vec.Word2Vec(rec, size=300,min_count=1)
model_name = "300features"
model.save(model_name)
model.syn0.shape
np_600=[]
for i in range(train.shape[0]):
    np1=np.array(model[train["Person"][i]])
    np2=np.array(model[train["Profession"][i]])
    #np3=np.array(model[train["cluster"][i]])
    arr1=np.append(np1,np2)
    #arr=np.append(arr1,np3)
    np_600.append(arr1)
    
train_data=pd.DataFrame.from_records(np_600)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(tr_clust)
list(le.classes_)
le.transform(tr_clust) 
#list(le.inverse_transform([2, 2, 1]))
train_data['cluster']=le.transform(tr_clust) 
#train_data['Profession']=train['Profession']

#pro_vectors = pd.read_csv( "C:/triple-scoring/features_profession.txt", quoting=3)

#for p in range(len(pro_vectors['Profession'])):
#    pro_vectors['Profession'][p]=re.sub("[^a-zA-Z0-9]", "", pro_vectors['Profession'][p])
    
#train_all=train_set
#for r in range(len(train_set['Profession'])):
#    for k in range(len(pro_vectors['Profession'])) :
#        if train_set['Profession'][r]==pro_vectors['Profession'][k]:
#            print(r)
#            pd.concat(train_all[r],pro_vectors[str(k)])

#train_data=pd.DataFrame.from_records(np.array(train_all))


#train_data=pd.merge(train_set,pro_vectors,how='inner',on='Profession')
#valid_data=train_data[450:515]
#valid_data_lab=train['Score'][450:515]
#train_data['Score']=train['Score'] 


#Define input array with angles from 60deg to 300deg converted to radians
#os.chdir("C:/triple-scoring")

#train = pd.read_csv('SAheart.csv')
#target = 'age'
#predictors = [x for x in train.columns if x not in [target,'famhist','chd']]
#Define the alpha values to test
#alpha_lasso = [1e-15,1e-12,1e-10,1e-5,1e-3]

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

param_test1 = [{'n_estimators':[100],'max_depth':[10],'learning_rate':[0.01]}]
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(max_features='sqrt',subsample=0.8), 
param_grid = param_test1, scoring='neg_mean_absolute_error',cv=5)
alg=gsearch1.fit(train_data[1:400],train['Score'][1:400])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_      

gbm_reg_dtrain_predictions = alg.predict(train_data[401:515])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
for d in range(len(gbm_reg_dtrain_predictions)):
       gbm_reg_dtrain_predictions[d]=math.floor(gbm_reg_dtrain_predictions[d])
        #print ("max score:",max(dtrain_predictions))

#for i in range(450,515)):
#    print(str(train['Score'][i])+str(":")+str(gbm_reg_dtrain_predcitions[i-450]))
print(get_accuracy(train['Score'][401:515],gbm_reg_dtrain_predictions))
print(get_distance(train['Score'][401:515],gbm_reg_dtrain_predictions))


# LASSO
from sklearn.linear_model import Lasso
param_test1 = [{'alpha':[1e-5,0.1,1,5]}]
gsearch1 = GridSearchCV(estimator = Lasso( fit_intercept=True,max_iter=100000), 
param_grid = param_test1, scoring='neg_mean_absolute_error',cv=5)
alg=gsearch1.fit(train_data[1:400],train['Score'][1:400])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_      

lasso_dtrain_predictions = alg.predict(train_data[401:515])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
for d in range(len(lasso_dtrain_predictions)):
        lasso_dtrain_predictions[d]=int(lasso_dtrain_predictions[d])
        #print ("max score:",max(dtrain_predictions))

print(get_accuracy(train['Score'][401:515],lasso_dtrain_predictions))
print(get_distance(train['Score'][401:515],lasso_dtrain_predictions))


# GBM classifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

param_test1 = [{'n_estimators':[100],'max_depth':[10],'learning_rate':[0.05]}]
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(max_features='sqrt',subsample=0.8), 
param_grid = param_test1, scoring='accuracy',cv=10)
alg=gsearch1.fit(train_data[1:400],train['Score'][1:400])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_      

gbm_cls_dtrain_predictions = alg.predict(train_data[401:515])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
for d in range(len(gbm_cls_dtrain_predictions)):
        gbm_cls_dtrain_predictions[d]=math.floor(gbm_cls_dtrain_predictions[d])
        #print ("max score:",max(dtrain_predictions))

print(get_accuracy(train['Score'][401:515],gbm_cls_dtrain_predictions))
print(get_distance(train['Score'][401:515],gbm_cls_dtrain_predictions))

np.corrcoef(gbm_reg_dtrain_predictions, gbm_cls_dtrain_predictions)[0, 1]
#np.corrcoef(lasso_dtrain_predictions, gbm_reg_dtrain_predictions)
preds=[]
for v in range(len(gbm_reg_dtrain_predictions)):
    preds.append(math.ceil((0.7*gbm_reg_dtrain_predictions[v]+0.3*gbm_cls_dtrain_predictions[v])))

print(get_accuracy(train['Score'][401:515],preds))
print(get_distance(train['Score'][401:515],preds))


















