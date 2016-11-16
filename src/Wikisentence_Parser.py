# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:49:10 2016

@author: Jay
"""

import pandas as pd
import numpy as np
import re
from itertools import chain
import os
import math
import itertools
from nltk.corpus import stopwords

cwd = os.getcwd()

def Content_process( sentence ):
    
    while True:
                start = sentence.find("[")
                middle = sentence.find("|")
                end = sentence.find("]")
                if start == -1 or middle == -1 or end == -1:
                    break
                if end <= start:
                    sentence = sentence[:end] + sentence[end+1:]
                    continue
                sentence = sentence[:start] + sentence[start+1: middle] + sentence[end+1:]
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z0-9_]", " ", sentence) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
    
    
##############################################################################################
##############  generating 200 profession files from wiki sentences ##########################
##############################################################################################
    
## delimiter new line is used ,it causes error bcz some lines have two \n so those lines are considered bad and removed 
## tab delimiter doesn't has this issue but lines are not well seperated.
#### frequent patterns with length 2,3,4 may have the n-gram person,profession combination
## this profession files may even furthur diveded based on person
os.chdir("C:/triple-scoring/Profession_sentences")    
df=pd.read_csv("C:/triple-scoring/wiki-sentences",sep='\n',header=None,nrows=500000,encoding='utf-8',error_bad_lines=False)   # remove nrows to run on the whole wikisentences file
df.columns=['Sentence']  
pro_df=pd.read_csv("C:/triple-scoring/professions",sep='\n',header=None,encoding='utf-8')
pro_df.columns=['Profession'] 
for i in range(pro_df.shape[0]):
    sentences_pro=[]
    profession=pro_df['Profession'][i]
    for j in range(df.shape[0]):
        sentence=df['Sentence'][j]  #.replace("_"," ")
        sentence_processed=Content_process(sentence)
        if sentence.find(profession)!=-1:
            sentences_pro.append(sentence_processed)
    print(len(sentences_pro))  
    sentences_df=pd.DataFrame(sentences_pro)
    sentences_df.to_csv(str(profession)+"_"+str("sentences.txt"),index_label=False,header=None,index=False)
    
    
    
#################################################################################################
############### generating 100 nation files from wiki sentences #################################
#################################################################################################
os.chdir("C:/triple-scoring/Nation_sentences")    
Nation_df=pd.read_csv("C:/triple-scoring/nationalities",sep='\n',header=None,encoding='utf-8')
Nation_df.columns=['Nationality'] 
for i in range(Nation_df.shape[0]):
    sentences_nation=[]
    nation=Nation_df['Nationality'][i]
    for j in range(df.shape[0]):
        sentence=df['Sentence'][j]  #.replace("_"," ")
        sentence_processed=Content_process(sentence)
        if sentence.find(nation)!=-1:
            sentences_nation.append(sentence_processed)
    print(len(sentences_nation))  
    sentences_df=pd.DataFrame(sentences_nation)
    sentences_df.to_csv(str(nation)+"_"+str("sentences.txt"),index_label=False,header=None,index=False)

##############################################################################################
##############  generating 134 person files in train from wiki sentences #####################
##############################################################################################
os.chdir("C:/triple-scoring/Person_sentences")  
train = pd.read_csv( "C:/triple-scoring/profession.train", header=None, 
delimiter="\t", quoting=3 ,encoding='utf-8')
train.columns=['Person','Profession','Score']     
unique_per=np.unique(train['Person']) 
for i in range(len(unique_per)):
    sentences_per=[]
    person=unique_per[i]
    for j in range(df.shape[0]):
        sentence=df['Sentence'][j]  #.replace("_"," ")
        sentence_processed=Content_process(sentence)
        if sentence.find(person)!=-1:
            sentences_per.append(sentence_processed)
    print(len(sentences_per))  
    sentences_df=pd.DataFrame(sentences_per)
    sentences_df.to_csv(str(person)+"_"+str("sentences.txt"),index_label=False,header=None,index=False)
    
    
    
##################################################################################################
#################  tf-idf for profession on the profession train file ############################
##################################################################################################
os.chdir("C:/triple-scoring/Profession_sentences") 
#pro_sen_df=pd.DataFrame(pro_df['Profession'])
from sklearn.feature_extraction.text import TfidfVectorizer

#### function to split words but not used #######
#def sentence_split( sentence ):
#    all_words=[]
#    for i in range(len(sentence)):
#        words = sentence[i].split(' ') 
#        all_words.append(words)
#    merged = list(itertools.chain(*all_words))
#    #sen_words=np.asarray(merged)             
#    return(merged)  
#pro_sen=[]

sen_all=[]
for p in range(pro_df.shape[0]):
    pro_file=pd.read_csv(str(pro_df['Profession'][p])+str("_")+str("sentences.txt"))
    for k in range(train.shape[0]):
        if train['Profession'][k]==pro_df['Profession'][p]:
            pro_file_np=np.array(pro_file,dtype=str)
    #pro_sen.append(pro_file_np)
    #np_sen=np.asarray(pro_sen)
    np_sen=' '.join(pro_file_np.flatten()) 
    #np_sen_processed=sentence_split(np_sen)
    #np_sen_rec=np.asarray(np_sen_processed,dtype=str)
    sen_all.append(np_sen)
    
## tf-idf
vectorizer = TfidfVectorizer(max_features=2000,min_df=1)
vec=vectorizer.fit_transform(sen_all)
features=np.matrix(vec.todense())
Pro_tfidf=pd.DataFrame(features)
Pro_tfidf['Profession']=pro_df['Profession']
os.chdir("C:/triple-scoring") 
Pro_tfidf.to_csv('Pro_tfidf.txt',sep="\t")            
    
     
    