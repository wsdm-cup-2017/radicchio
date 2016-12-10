'''
The only restriction is that we cannot manually label more training data ..
So for from a sample of say 300 persons (not there in the training data, for each person word vector
find the distance from that person to all the 200 nationalities and professions...
Then find min and max distance for a person - nationaity ( or profession) pair.
Then, do min max scaling derive the relavence score for a few more pairs, and classify the given tuple
with the relavence score (based on which band it falls under... save these bands in a list of lists each sub list is size 2 (min and max)
and there are 7 such ranges, for each person - nationality (or profession ) pair.

then use these relavence scores as extra data in training the models...
plus train more models too... (like include random forest regression... only regression models will work??
and train on the pairwise embeddings, since this is what will be given...
(create more pairwise embeddings for the new person - profession (or nationality ) pairs....

'''

import random
import pandas as pd
import re
import copy
import numpy as np

path = "C:\\Sharan\\CS 412\\cs 412 proj\\7th 8th\\radicchio-399581f9006c39a5b4a2dc0a91f818db3a7bc398\\data\\"
fil = path + "persons.txt"
persons_df = pd.read_csv(fil, sep='\t',header=None)
persons_df.columns=['Person','Free base id']
persons=list(persons_df['Person'])

fil = path + "professions"
professions_df = pd.read_csv(fil, sep='\n',header=None)
professions_df.columns=['profession']
professions=list(professions_df['profession'])


fil = path + "nationalities"
nationalities_df = pd.read_csv(fil, sep='\n',header=None)
nationalities_df.columns=['nationality']
nationalities=list(nationalities_df['nationality'])

############################
#getting the list of persons alteady there in the old train files

fil = path + "profession.train"
persons_prof_df = pd.read_csv(fil, sep='\t',header=None)
persons_prof_df.columns=['Person','profession','score']
persons_prof=list(set(persons_prof_df['Person']))

fil = path + "nationality.train"
persons_nation_df = pd.read_csv(fil, sep='\t',header=None)
persons_nation_df.columns=['Person','nationality','score']
persons_nat=list(set(persons_nation_df['Person']))

#########################

#so first inputting the 2 kb files
fil = path + "nationality.kb.txt"
nationalities_kb_df = pd.read_csv(fil, sep='\t',header=None)
nationalities_kb_df.columns=['person', 'nationality']
nationalities_kb=[]
nationalities_kb.append(list(nationalities_kb_df['person']))
nationalities_kb.append(list(nationalities_kb_df['nationality']))
'''
for i in range(0,len(nationalities_kb_df['person']),1):
    row=[]
    row.append(nationalities_kb_df['person'].iloc[i])
    row.append(nationalities_kb_df['nationality'].iloc[i])
    nationalities_kb.append(row)
'''

#so first inputting the 2 kb files
fil = path + "profession.kb.txt"
professions_kb_df = pd.read_csv(fil, sep='\t',header=None)
professions_kb_df.columns=['person', 'nationality']
professions_kb=[]
professions_kb.append(list(professions_kb_df['person']))
professions_kb.append(list(professions_kb_df['nationality']))

'''
for i in range(0,len(professions_kb_df['person']),1):
    row=[]
    row.append(professions_kb_df['person'].iloc[i])
    row.append(professions_kb_df['nationality'].iloc[i])
    professions_kb.append(row)
'''


##################
#TO FIND THE NEW PESONS TO BE ADDED TO THE LIST


subsetlist=random.sample(persons,300)
persons_p_new=[]
persons_n_new=[]
#new persons to be added onto the professions
#new persons to be added ont the natinality list
for i in range(0, len(subsetlist),1):
    if (subsetlist[i] in persons_prof) ==False:
        persons_p_new.append(subsetlist[i])

    if (subsetlist[i] in persons_nat ) == False:
        persons_n_new.append(subsetlist[i])


###############################################################################

#we need to normalize the stuff... lowercase and remove punctuations,
def normalize(text):
	text  = re.sub(r'[^\w\s]' , " ", text, re.UNICODE)#remove punctuations
	text = text.lower()
	#terms = text.split()# the splitting will be done later
	return text

variables=['persons_p_new','persons_n_new','nationalities','professions']

for v in variables:
    for i in range(0,len(globals()[v]),1):
        globals()[v][i]=normalize(globals()[v][i])



'''
#first we convert all the files into lower case..
persons_p_new=[a.lower() for a in persons_p_new]
persons_n_new=[a.lower() for a in persons_n_new]
nationalities=[a.lower() for a in nationalities]
professions=[a.lower() for a in professions]
'''

#########################################################################

#GETTING WORD VECTORS for the new PERSONS, the NATIONALITIES and PROFESSIONS
# now we just get the respective word vectors for all the nationalities, professions and new persons
#word_vecs=[]
nat_vec=[]
prof_vec=[]
newpersons_p_vec=[]
newpersons_n_vec=[]

zerovec=[0.0]*301

variables2=['nat_vec','prof_vec','newpersons_p_vec','newpersons_n_vec']
variables1=['nationalities','professions','persons_p_new','persons_n_new']

for v in range(0,4,1):
    for i in range(0,len(globals()[variables1[v]]),1):
        globals()[variables2[v]].append(zerovec)


nat_vec=np.array(nat_vec)
prof_vec=np.array(prof_vec)
newpersons_p_vec=np.array(newpersons_p_vec)
newpersons_n_vec=np.array(newpersons_n_vec)

# The file is too big to load in memory...
#so we just take it line by line, compare and update the respective vector

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

        for v in range(0,4,1):
            for i in range(0, len(globals()[variables1[v]]), 1):#ERROR = THIS FOR LOOOOP MESSES SOME ADDITION UP!?!?!? BUT ONE ITERATON OF THIS LOOP IS CORRECT!
                #there are 4 occurances of the term 'activist' in the professions file... but in different i's...
                #but, for each i, it updates it 4 times!!!
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

#461048 words in the word vector file....
#461048 word vectors!!

#now when we come out we need to divide each weight by the count in corresponding row
#then delete the count column

#it is possible that some persons dont have word vectors.... such persons are just removed if the count is 0
for v in range(0,4,1):
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

#NOW WE HAVE THE  WEIGHT VECTORS for the words in nationality, profession and person!...
#the following are the variables that have the final data.
#variables2=['nat_vec','prof_vec','newpersons_p_vec','newpersons_n_vec']
#variables1=['nationalities','professions','persons_p_new','persons_n_new']


###########################

#THESE ARE PERSON - nat (or prof) pairs for the new persons got from the .kb files
#for the new persons to be added we need to find the corresponding professions and nationalities from the kb files....
#and make a list of persons and professions

#first we normalize the kb file
for i in range(0, len(professions_kb[0]), 1):
    professions_kb[0][i] = normalize(professions_kb[0][i])
    professions_kb[1][i] = normalize(professions_kb[1][i])

for i in range(0, len(nationalities_kb[0]), 1):
    nationalities_kb[0][i] = normalize(nationalities_kb[0][i])
    nationalities_kb[1][i] = normalize(nationalities_kb[1][i])


#indices = [i for i, x in enumerate(nationalities_kb[0]) if x == 'Abhimanyu Rajp']
persons_prof_new=[]
persons_nat_new=[]
for j in range(0, len(persons_p_new),1):
    indices = [i for i, x in enumerate(professions_kb[0]) if x == persons_p_new[j]]
    for k in indices:
        row=[]
        row.append(professions_kb[0][k])
        row.append(professions_kb[1][k])
        persons_prof_new.append(row)

for j in range(0, len(persons_n_new), 1):
    indices = [i for i, x in enumerate(nationalities_kb[0]) if x == persons_n_new[j]]
    for k in indices:
        row = []
        row.append(nationalities_kb[0][k])
        row.append(nationalities_kb[1][k])
        persons_nat_new.append(row)

########################################################################################
# Now we find DISTANCE between each person - nationality and person - profession pair! #
########################################################################################

from scipy.spatial import distance

# we now find the distance between a person 's vector and every nationality and profession...

variables2=['nat_vec','prof_vec','newpersons_p_vec','newpersons_n_vec'] #np arrays with the distance measures
variables1=['nationalities','professions','persons_p_new','persons_n_new'] # the list containing the names of the ppeople

per_prof_score=[]
#First people and nationalities
for i in range(0,len(persons_p_new),1):
    combo=[]
    combo.append([])#person
    combo.append([])#profession
    combo.append([])#eucleadean distance
    # one idea can be to add more distance measures, and take a majority vote of the predicted class from that...
    #(needs implementation)
    score_ranges=[]
    #now for each person we compute the distance between ethe person and each of the professions.
    for j in range(0,len(professions),1):
        combo[0].append(persons_p_new[i])
        combo[1].append(professions[j])
        dist=distance.cosine(newpersons_p_vec[i],prof_vec[j])
        combo[2].append(dist)

    minimum=min(combo[2])
    maximum=max(combo[2])
    common_d=(maximum-minimum)/8
    for k in range(0,8,1):
        #the closer it is the higher the relavence score.... so we start from maximum and assign that to 0..
        lowerlimit = maximum - (k+1)*common_d
        upperlimit = maximum - (k)*common_d
        sc_rng=[]
        sc_rng.append(lowerlimit)
        sc_rng.append(upperlimit)
        score_ranges.append(sc_rng)

    #now we check for in the person -profession file which range it falls within... for this person.. and append the predicted relavence score
    indices = [p for p, x in enumerate(professions_kb[0]) if x == persons_p_new[i]]
    for indi in indices:
        nat_index = combo[1].index(professions_kb[1][indi])
        #now we just see where the corresponding distance for this person and nationality combo falls in
        score=0
        for j in range(0,8,1):
            if score>=score_ranges[j][0] and score<=score_ranges[j][1]:
                break #so the j value is the predicted score

        #now we just append to the final output list
        row=[]
        row.append(persons_p_new[i])
        row.append(combo[1][nat_index])
        row.append(j)

        per_prof_score.append(row)


############### now we find for nationalitites


per_nat_score=[]
#First people and nationalities
for i in range(0,len(persons_n_new),1):
    combo=[]
    combo.append([])#person
    combo.append([])#nationality
    combo.append([])#eucleadean distance
    # one idea can be to add more distance measures, and take a majority vote of the predicted class from that...
    #(needs implementation)
    score_ranges=[]
    #now for each person we compute the distance between ethe person and each of the professions.
    for j in range(0,len(nationalities),1):
        combo[0].append(persons_n_new[i])
        combo[1].append(nationalities[j])
        dist=distance.cosine(newpersons_n_vec[i],nat_vec[j])
        combo[2].append(dist)

    minimum=min(combo[2])
    maximum=max(combo[2])
    common_d=(maximum-minimum)/8
    for k in range(0,8,1):
        #the closer it is the higher the relavence score.... so we start from maximum and assign that to 0..
        lowerlimit = maximum - (k+1)*common_d
        upperlimit = maximum - (k)*common_d
        sc_rng=[]
        sc_rng.append(lowerlimit)
        sc_rng.append(upperlimit)
        score_ranges.append(sc_rng)

    #now we check for in the person -profession file which range it falls within... for this person.. and append the predicted relavence score
    indices = [p for p, x in enumerate(nationalities_kb[0]) if x == persons_n_new[i]]
    for indi in indices:
        nat_index = combo[1].index(nationalities_kb[1][indi])
        #now we just see where the corresponding distance for this person and nationality combo falls in
        score=0
        for j in range(0,8,1):
            if score>=score_ranges[j][0] and score<=score_ranges[j][1]:
                break #so the j value is the predicted score

        #now we just append to the final output list
        row=[]
        row.append(persons_n_new[i])
        row.append(combo[1][nat_index])
        row.append(j)

        per_nat_score.append(row)


#per_prof_score and per_nat_score are the final output of this file....

#####################################
#OUTPUTTING results to file:
####################################

'''
# need to take care of encoding before writing to file

clsdd = "per-nat-score"
op = open(path + clsdd + ".txt", "w")
for i in range(0,len(per_nat_score),1):
    op.write(str(per_nat_score[i][0]) + " " + str(per_nat_score[i][1]) + " " + str(per_nat_score[i][2])+ "\n")

clsdd = "per-prof-score"
op = open(path + clsdd + ".txt", "w")
for i in range(0, len(per_prof_score), 1):
    op.write(per_prof_score[i][0] + " " + per_prof_score[i][1] + " " + str(per_prof_score[i][2]) + "\n")


'''