# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:42:17 2016

@author: Jay
"""

import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
#from gensim.models import word2vec
#from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS

#from itertools import chain

#####################################################
# Exploring profession train dataset ################
#####################################################
pr_df = pd.read_csv("C:/triple-scoring/profession.train", sep='\t',header=None)
pr_df.columns=['Person','Profession','Score']
#triple_df['Person']

pr_df.head(3)

scr_grp=pr_df.groupby(['Score'])
scr_grp.size()
plt.hist(pr_df['Score'])
plt.title('score histogram for profession train')

prof_grp=pr_df.groupby(['Profession'])
max(prof_grp.size())
print("number of professions in train" , len(prof_grp.size()))

# word cloud for profession based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(pr_df['Profession']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

per_grp=pr_df.groupby(['Person'])
per_grp.size()
print("number of persons in train" , len(per_grp.size()))

# word cloud for person based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(pr_df['Person']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#####################################################
# Exploring profession.kb  dataset ################
#####################################################
pr_kb_df = pd.read_csv("C:/triple-scoring/profession.kb", sep='\t',header=None)
pr_kb_df.columns=['Person','Profession']
print("number of observations in profession kb file", pr_kb_df.shape[0])

prof_kb_grp=pr_kb_df.groupby(['Profession'])
prof_kb_grp.size()
print("number of professions in profession KB" , len(prof_kb_grp.size()))

# word cloud for profession KB based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(pr_kb_df['Profession']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


per_kb_grp=pr_kb_df.groupby(['Person'])
per_kb_grp.size()
print("number of persons in profession KB" , len(per_kb_grp.size()))

# word cloud for Person KB based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(pr_kb_df['Person']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

######################################################
### Exploring nationality train dataset ##############
######################################################

na_df = pd.read_csv("C:/triple-scoring/nationality.train", sep='\t',header=None)
na_df.columns=['Person','Nationality','Score']
#triple_df['Person']

na_df.head(3)

na_scr_grp=na_df.groupby(['Score'])
na_scr_grp.size()
plt.hist(na_df['Score'])
plt.title('score histogram for nationality train')

na_grp=na_df.groupby(['Nationality'])
na_grp.size()
print("number of nationalities in train" , len(na_grp.size()))

# word cloud for Nationality based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(na_df['Nationality']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

na_per_grp=na_df.groupby(['Person'])
na_per_grp.size()
print("number of persons in train" , len(na_per_grp.size()))

# word cloud for Person based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(na_df['Person']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

######################################################
### Exploring nationality KB dataset ################
######################################################

na_kb_df = pd.read_csv("C:/triple-scoring/nationality.kb", sep='\t',header=None)
na_kb_df.columns=['Person','Nationality']
print("number of observations in profession kb file", na_kb_df.shape[0])
na_kb_df.head(3)

na_kb_grp=na_kb_df.groupby(['Nationality'])
na_kb_grp.size()
print("number of nationalities in train" , len(na_kb_grp.size()))

# word cloud for Nationality KB based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(na_kb_df['Nationality']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

na_kb_per_grp=na_kb_df.groupby(['Person'])
na_kb_per_grp.size()
print("number of persons in train" , len(na_kb_per_grp.size()))

# word cloud for Person based on frequency
wordcloud = WordCloud(max_font_size=50).generate(str(na_kb_df['Person']))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

########################################
#### clustering based on tfidf way. ####
########################################
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(min_df=1)
corpus=[]
corpus1=pr_df['Person'].tolist()
corpus2=pr_df['Profession'].tolist()
#corpus.append(pr_df['Profession'].tolist())
tfidf_matrix1 =  tf.fit_transform(corpus1)
tfidf_matrix2 =  tf.fit_transform(corpus2)

feature_names = tf.get_feature_names()

