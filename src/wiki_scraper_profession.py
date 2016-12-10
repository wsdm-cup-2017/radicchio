# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:07:42 2016

@author: Jay
"""

import wikipedia
import re
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords # Import the stop word list
#print (stopwords.words("english")) 


# currently scraping for each person not combination of profession if person works fine then create fetures for the combination

os.chdir("C:/triple-scoring")

train = pd.read_csv( "C:/triple-scoring/professions.txt", header=None, 
 delimiter="\t", quoting=3 )
train.columns=['Profession'] 
wiki_rec=[]

def Content_process( raw_review ):
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z0-9]", " ", content) 
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

for i in range(len(train['Profession'])):
    wiki = wikipedia.search(str(train['Profession'][i]),results=10000)
    
    content=' '.join(wiki)
    processed_content= Content_process(content)
    #rec=np.array(processed_content.split(' '))
    wiki_rec.append(processed_content)

#clean_wiki_reviews = []
#for i in range(len(wiki_rec)):
#    # If the index is evenly divisible by 1000, print a message                                                               
#    clean_wiki_reviews.append(wiki_rec[i].split(' '))    

len(wiki_rec)
#len(train['Person'])

wiki_records=np.asarray(wiki_rec)
#train['wiki']=wiki_records
#import os
#cwd = os.getcwd()
#train.to_csv('pr_wiki.txt',columns=('Profession','wiki'),sep="\t")




#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(analyzer = "word",   \
#                             tokenizer = None,    \
#                             preprocessor = None, \
#                             stop_words = None,   \
#                             max_features = 1000) 
#
## fit_transform() does two functions: First, it fits the model
## and learns the vocabulary; second, it transforms our training data
## into feature vectors. The input to fit_transform should be a list of 
## strings.
#train_data_features = vectorizer.fit_transform(wiki_records.ravel())
#
## Numpy arrays are easy to work with, so convert the result to an 
## array
#train_data_features = train_data_features.toarray()
#vocab = vectorizer.get_feature_names()
#print (len(vocab))

##################################
### tf-idf tokenization ###########
################################

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000,min_df=1,ngram_range=(1))
vec=vectorizer.fit_transform(wiki_records)
features=np.matrix(vec.todense())

df=pd.DataFrame(features)
df['Profession']=train['Profession']
df.to_csv('features_profession.txt',header=True)

# hierarchical clustering
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


Z = hierarchy.linkage(features,'average')
plt.figure()
dn = hierarchy.dendrogram(Z)
#dn1 = hierarchy.dendrogram(Z, ax=train['cluster'],orientation='top')





   
    
    
       