"""
Utility functions for reading freebase features.
"""

import os

def filter_freebase_features(
        freebase_feature_filename, persons_filename, freebase_mapping_filename):
    """
    Filters freebase features for the given training files.
    """
    freebase_file = open(freebase_feature_filename, 'r')
    persons_file = open(persons_filename, 'r')
    freebase_id_mapping_file = open(freebase_mapping_filename, 'r')

    person_names = set([x.strip().split('\t')[0] for x in persons_file.xreadlines()])

    freebase_id_reversemap = {}
    for freebase_id_line in freebase_id_mapping_file.xreadlines():
        parts = freebase_id_line.strip().split('\t')

        if parts[0] in person_names:
            freebase_id_reversemap[parts[1]] = parts[0]

    features = {}
    for feature_line in freebase_file.xreadlines():
        parts = feature_line.strip().split('\t')

        if parts[1] in freebase_id_reversemap:
            features[freebase_id_reversemap[parts[1]]] = parts[2]

    return features

if __name__ == "__main__":
    entity_notes_features_file = "../../entity-ontos.txt/entity-ontos.txt"
    persons_training_file = "../data/profession.train"
    freebase_mapping_file = "../../triple-scoring/persons"
    output_file = open("freebase_features.txt", 'w')

    freebase_features = filter_freebase_features(entity_notes_features_file,
        persons_training_file, freebase_mapping_file)

    for (key, value) in freebase_features.iteritems():
        output_file.write(key + "\t" + value + os.linesep)

    output_file.close()
