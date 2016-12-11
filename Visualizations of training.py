
import random
import pandas as pd
import re
import copy
import numpy as np



############################
#getting the list of persons alteady there in the old train files
path = "C:\\Sharan\\CS 412\\cs 412 proj\\7th 8th\\radicchio-399581f9006c39a5b4a2dc0a91f818db3a7bc398\\data\\"

fil = path + "profession.train"
persons_prof_df = pd.read_csv(fil, sep='\t',header=None)
persons_prof_df.columns=['Person','profession','score']
persons_prof=list(set(persons_prof_df['Person']))

fil = path + "nationality.train"
persons_nation_df = pd.read_csv(fil, sep='\t',header=None)
persons_nation_df.columns=['Person','nationality','score']
persons_nat=list(set(persons_nation_df['Person']))

#########################


#we need to normalize the stuff... lowercase and remove punctuations,
def normalize(text):
	text  = re.sub(r'[^\w\s]' , " ", text, re.UNICODE)#remove punctuations
	text = text.lower()
	#terms = text.split()# the splitting will be done later
	return text

variables1=['persons_prof','persons_nat']

for v in variables1:
    for i in range(0,len(globals()[v]),1):
        globals()[v][i]=normalize(globals()[v][i])

#######

#this are arrays
persons_prof_vec=[]
persons_nat_vec=[]

variables2=['persons_prof_vec','persons_nat_vec']

zerovec=[0.0]*301

for v in range(0,2,1):
    for i in range(0,len(globals()[variables1[v]]),1):
        globals()[variables2[v]].append(zerovec)


persons_prof_vec=np.array(persons_prof_vec)
persons_nat_vec=np.array(persons_nat_vec)


fil = path + "vectors.txt"

c=0
# we go through each line (word) in the word vector file... then we search if that work is there in any of the files....
# if yes, we update the weights in each of the 4 files above
with open(fil) as infile:
    for line in infile:
        a = line.split(" ")
        '''
        print(a[0])
        print(a[1:])
        c+=1
        if c>2:
            break
        '''
        # we just convery the numerical string into float values.
        #and then append the line to one of the lists if relevent

        for v in range(0,2,1):
            for i in range(0, len(globals()[variables1[v]]), 1):
                temp=copy.deepcopy(globals()[variables1[v]][i])
                row=temp.split(" ")
                for rowele in row:
                    if a[0]==rowele:#now we add the weight vectors onto the corresponding row column 2..and increment counter in column 3
                        globals()[variables2[v]][i][300]+=1
                        print("i= " + str(i) + " and ")
                        for w in range(1,301,1):
                            float_a=float(a[w])
                            globals()[variables2[v]][i][w-1] +=  float_a

        c=c+1
        if c % 100 ==0:
            print(c)


for v in range(0,2,1):
    l=len(globals()[variables2[v]])
    i=0
    while(i<l):
        if globals()[variables2[v]][i][300]==0:
            globals()[variables2[v]] = np.delete(globals()[variables2[v]],(i),axis=0)
            del globals()[variables1[v]][i]
            l = len(globals()[variables2[v]])
            continue

        for w in range(0, 300, 1):
            globals()[variables2[v]][i][w] = globals()[variables2[v]][i][w] / float(globals()[variables2[v]][i][300])

        i=i+1

for v in variables2:
    globals()[v]=np.delete(globals()[v],(300),axis=1)

##########################################################################################
## now we just reduce the dimension of the persons vector from 300 d to 2 d and plot it ##
##########################################################################################
#from scikit-learn.manifold import tsne

#matplot lib renders the output through some other package.... and in this case it was looking for PyQt5 which is not there on my machone...
#so when u change the backend rendering package that matplotlib uses to TkAgg(ie matplotlib.use(TkAgg))
#it then is able to use matplot lib

#matplotlib.use('TkAgg')

import matplotlib
import matplotlib.pyplot as plt#error
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter


from sklearn import manifold

X=persons_prof_vec
n_components = 2

#2 dimentios
#fit transform
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)


import numpy as np
import matplotlib.pyplot as plt
import random

N = len(Y)
data = Y
labels = persons_prof
plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    data[:,0], data[:, 1], marker = 'o',
    cmap = plt.get_cmap('Spectral'))


'''
#RANDOM SUBSETTING:
#now we label the points... we need to label only one in 4 or 5 points... not all
subset=[]
for i in range(0,N,1):
    subset.append(i)

subsetlist=random.sample(subset,round(len(subset)/4))

subsetdataX=[]
subsetdataY=[]
subsetlabels=[]

for i in range(0,len(subsetlist),1):
    subsetdataX.append(Y[subsetlist[i]][0])
    subsetdataY.append(Y[subsetlist[i]][1])
    subsetlabels.append(labels[subsetlist[i]])
'''


#ONLY SUBSETTING THOSE which are of same profession and score combo (just change the if condition>>>)
per_list=list(persons_prof_df['Person'])
prof_list=list(persons_prof_df['profession'])
score_list=list(persons_prof_df['score'])

indices=[]
for i in range(0,len(per_list),1):
    #if prof_list[i]=="Author":# and score_list[i]==7:
    if prof_list[i] == "Author" and score_list[i] == 7:

        indices.append(i)

subsetlist=[]
for p in indices:
    normalized_per=normalize(per_list[p])
    ind=persons_prof.index(normalized_per)
    subsetlist.append(ind)

subsetlist=list(set(subsetlist))

subsetdataX=[]
subsetdataY=[]
subsetlabels=[]

for i in range(0,len(subsetlist),1):
    subsetdataX.append(Y[subsetlist[i]][0])
    subsetdataY.append(Y[subsetlist[i]][1])
    subsetlabels.append(labels[subsetlist[i]])

#for label, x, y in zip(labels, data[:, 0], data[:, 1]):
for label, x, y in zip(subsetlabels, subsetdataX, subsetdataY):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'grey', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()