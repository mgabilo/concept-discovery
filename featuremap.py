#!/usr/bin/python

def feature_to_str(feature):
    if isinstance(feature, tuple):
        return '%s %s' % (feature[0] , feature[1])
    return feature

def load_feature_map(filename='features.idx'):
    feature_map_toidx = {}
    feature_map_fromidx = []
    fin = open(filename)
    for n, line in enumerate(fin):
        line = line.strip()
        section, delim, feature = line.partition(' ')
        feature_map_toidx[ (section, feature) ] = n
        feature_map_fromidx.append( (section, feature) )
    fin.close()
    return (feature_map_toidx, feature_map_fromidx)

def load_concept_map(filename='concepts.idx'):
    concept_map_toidx = {}
    concept_map_fromidx = []
    fin = open(filename)
    for n, line in enumerate(fin):
        line = line.strip()
        concept_map_toidx[line] = n
        concept_map_fromidx.append(line)
    fin.close()
    return (concept_map_toidx, concept_map_fromidx)

def load_word_map(filename='words.idx'):
    word_map_toidx = {}
    word_map_fromidx = []
    fin = open(filename)
    for n, line in enumerate(fin):
        line = line.strip()
        word_map_toidx[line] = n
        word_map_fromidx.append(line)
    fin.close()
    return (word_map_toidx, word_map_fromidx)

