import numpy as np
from collections import defaultdict
from _SupervisedBase import SupervisedBase
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
"""
This file implements several supervised learning models that inherit from the SupervisedBase.
"""
class AllZeroFeature(SupervisedBase):
    """
    Generate 2-dimensional features for each instance.
    All the features are 0.
    """
    def extract_features(self, pairs):
        return np.zeros((len(pairs),2))

class WordVector(SupervisedBase):
    """
    Use word vectors as features.
    Currently, we only support loading fetaures from files.
    """
    def extract_features(self, pairs, load_from_X = True, X_path = "../data/X_name_profession_600.np"):
	scaler = StandardScaler()
	if load_from_X is True:
	    X = np.load(X_path)

	#standardly scale features
	X = scaler.fit_transform(X)
        return X

class FreebaseFeature(SupervisedBase):
    """
    Using Freebase based features
    """
    def __init__(self):
        SupervisedBase.__init__(self, learner_type="RandomForest", parameters={"n_estimators":100})

        freebase_features = defaultdict(str)
        feature_file = open('freebase_features.txt', 'r')

        professions = open("../../triple-scoring/professions", 'r')
        all_professions = set([x.strip() for x in professions.xreadlines()])

        for lines in feature_file.xreadlines():
            parts = lines.split('\t')
            freebase_features[parts[0]] = parts[1]
        self.freebase_raw_features = freebase_features

        tokenizer = lambda sentence: sentence.strip().split()
        self.feature_vectorizer = CountVectorizer(analyzer=tokenizer)
        self.feature_vectorizer.fit([self.append_profession(x, prof) for x in freebase_features.values() for prof in all_professions])

        print len(self.feature_vectorizer.get_feature_names())

        self.word2vec_model = WordVector()

    def extract_features(self, pairs):

        features = [self.append_profession(self.freebase_raw_features[name], profession) for (name, profession) in pairs]

        # more_features = self.word2vec_model.extract_features(pairs)

        # return np.concatenate((self.feature_vectorizer.transform(features).toarray(), more_features), axis=1)

        return self.feature_vectorizer.transform(features)

    def append_profession(self, feature_name, profession):
        escaped = profession.replace(' ', '_')
        return feature_name.replace(' ', '.' + escaped + ' ')

    def evaluate(self):
        SupervisedBase.evaluate(self)

        inverse_map = defaultdict(str)
        for key in self.feature_vectorizer.vocabulary_:
            inverse_map[self.feature_vectorizer.vocabulary_.get(key)] = key

        num_features = 100
        for feature_id in self.learner.feature_importances_.argsort()[-num_features:][::-1]:
            print str(feature_id) + " " + inverse_map[feature_id] + " " + str(self.learner.feature_importances_[feature_id])