#!/usr/bin/python
import features
import glob
from nltk.corpus import wordnet as nlwn
import sys
import pdb
import operator
import os
import pdb
import sqlite3
import simplejson as json
import math

def get_feature_set(source_dir):
    print 'Loading models...'
    pmi_models = []
    realcount  = 0
    feature_set = set()

    for n, pmi_filename in enumerate(glob.glob('%s/*.pmi.bz2' % source_dir)):
        print 'Loading:', n, pmi_filename
        pmi_model = features.PMINounModel()
        pmi_model._load_from_pmi_file(pmi_filename)
        for section in pmi_model.sections:
            for feature in pmi_model.__dict__[section]:
                if (section, feature) not in feature_set:
                    feature_set.add( (section, feature) )
    return feature_set

def save_feature_set(features, filename):
    fout = open(filename, 'w')
    for section, feature in features:
        if isinstance(feature, tuple):
            fout.write('%s %s %s\n' % (section, feature[0], feature[1]))
        else:
            fout.write('%s %s\n' % (section, feature))
    fout.close()

def main():
    source_dir = sys.argv[1]
    features_file = sys.argv[2]
    features = get_feature_set(source_dir)
    save_feature_set(features, features_file)

if __name__ == "__main__":
    main()
