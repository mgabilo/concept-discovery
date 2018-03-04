#!/usr/bin/python

import features
import operator
import pdb
import sys
import simplejson as json
import os
from nltk.corpus import wordnet as nlwn
import sqlite3
from make_pmi_db import load_concepts_mem_db
from make_pmi_db import load_words_mem_db
import featuremap
import math
from collections import defaultdict

def load_concept_model(concept):
    try:
        model = features.PMINounModel(concept, base_dir)
        return model
    except IOError:
        return None

def prune_concepts(current_synset, current_model=None):
    hyponym_list = []
    for hyponym in current_synset.hyponyms() + current_synset.instance_hyponyms():

        hypo_model = load_concept_model(hyponym.name())
        hyponym_list.append((hyponym, hypo_model))

        try:
            os.stat(hypo_model.lemma_to_filename(hyponym.name(), new_dir))
            hypo_model.load_from_pmi_file(new_dir, hyponym.name())
        except OSError:
            prune_concepts(hyponym, hypo_model)

    if current_model:

        union_max_list = []

        for hyponym, hypo_model in hyponym_list:
            if hypo_model:
                union_max_list.append(hypo_model)
                hypo_model.save_to_file(new_dir)

        for hypo_model in union_max_list:
            current_model.union_max(hypo_model)

def main():
    global base_dir
    base_dir = sys.argv[1]

    global new_dir
    new_dir = sys.argv[2]

    prune_concepts(nlwn.synset('entity.n.01'))

if __name__ == "__main__":
    main()
