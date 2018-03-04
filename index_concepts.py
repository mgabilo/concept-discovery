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

def get_concept_set():

    for concept in nlwn.all_synsets(pos='n'):
		yield concept.name()

def save_concept_set(concepts, filename):
    fout = open(filename, 'w')
    for concept in concepts:
        fout.write('%s\n' % (concept,))
    fout.close()

def main():
    concepts_file = sys.argv[1]
    concepts = get_concept_set()
    save_concept_set(concepts, concepts_file)

if __name__ == "__main__":
    main()
