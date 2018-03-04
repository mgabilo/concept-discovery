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
from bz2 import BZ2File

def get_feature_set(source_dir):
    print 'Loading models...'
    pmi_models = []
    realcount  = 0
    feature_set = set()

    for n, pmi_filename in enumerate(glob.glob('%s/*.pmi.bz2' % source_dir)):
        print 'Loading:', n, pmi_filename
        fin = BZ2File(pmi_filename)
        word = fin.readline().strip()
        fin.close()
        if word not in feature_set:
            feature_set.add(word)
    return feature_set

def save_feature_set(features, filename):
    fout = open(filename, 'w')
    for word in features:
        fout.write('%s\n' % (word,))
    fout.close()

def main():
    source_dir = sys.argv[1]
    features_file = sys.argv[2]
    features = get_feature_set(source_dir)
    save_feature_set(features, features_file)

if __name__ == "__main__":
    main()
