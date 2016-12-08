"""
Utility functions for reading freebase features.
"""

import shelve
import cPickle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

def gen_freebase_features(
        freebase_feature_filename, freebase_save):
    """
    Filters freebase features for the given training files.
    """

    tokenizer = lambda sentence: sentence.strip().split()
    feature_vectorizer = CountVectorizer(tokenizer=tokenizer)
    known_features = set()

    with open(freebase_feature_filename, 'r') as freebase_file:
        for feature_line in freebase_file.xreadlines():
            parts = feature_line.strip().split('\t')

            name = parts[0]
            freebase_id = parts[1]
            freebase_features = parts[2] # Space separated

            for feat in freebase_features.strip().split():
                known_features.add(feat)

    feature_vectorizer.fit(list(known_features))

    persistence_store = shelve.open(freebase_save)
    with open(freebase_feature_filename, 'r') as freebase_file:
        for feature_line in freebase_file.xreadlines():
            parts = feature_line.strip().split('\t')

            freebase_id = parts[1]
            freebase_features = parts[2] # Space separated
            persistence_store[freebase_id] = feature_vectorizer.transform([freebase_features.strip()])

    cPickle.dump(feature_vectorizer.vocabulary_, open('freebase_feature_mapping.pkl', 'w'))
    persistence_store.close()


if __name__ == "__main__":
    entity_notes_features_file = "../../entity-ontos.txt/entity-ontos.txt"
    freebase_features = gen_freebase_features(entity_notes_features_file,
        'freebase_save.bin')
