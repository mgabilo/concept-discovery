#!/usr/bin/python

# word_concept_topsim.py
# * requires words and concepts sqlitedb that describes the features of each
# word and concept.
# * generate the top similar list of concepts for each word.
# * groupbyconcepts.py finds the inverse of the generated file.

import features
import operator
import pdb
import sys
import simplejson
import os
from nltk.corpus import wordnet as nlwn
import sqlite3
from make_pmi_db import load_concepts_mem_db
from make_pmi_db import load_words_mem_db
import featuremap
import math

ASSIGN_COMMITTEES_T = 0.045

word_map_toidx, word_map_fromidx = featuremap.load_concept_map('words.idx')
concept_map_toidx, concept_map_fromidx = featuremap.load_concept_map()
feature_map_toidx, feature_map_fromidx = featuremap.load_feature_map()

def sim(word, sim_c, concept1_dict, concept2):
    sim_c.execute('''select feature, pmi from concepts where concept = ? and pmi > 6.0''', (concept2,))
    concept2_dict = {}
    for feature, pmi in sim_c:
        concept2_dict[feature] = pmi

    for feature in concept2_dict:
        if concept1_dict.has_key(feature):
            feature_name = feature_map_fromidx[feature]
            human_readable(feature_name, word)

def human_readable(feature_tup, word):
    section, feature = feature_tup
    if section == 'prep_head_noun_pairs':
        prep, np = feature.split()
        print section, ':', np, prep, word
    elif section == 'prep_head_noun_of_pp_pairs':
        prep, np = feature.split()
        print section, ':', word, prep, np
    elif section == 'prep_verb_pairs':
        prep, verb = feature.split()
        print section, ':', verb, prep, word
    elif section == 'object1_head_of_verbs':
        print section, ':', feature, word
    elif section == 'subject_head_of_verbs':
        print section, ':', word, feature

    #elif section == 'modifiers_of_head':
    #    pos, mod = feature.split()
    #    print section, ':', mod, word

    elif section == 'modified_heads':
        print section, ':', word, feature

def view_common(word, concept, words_dbfilein='words-idx.db', concepts_dbfilein='cv-upper-level-no-parent-0.65-200-noparentfix.db'):

    #conn = load_words_mem_db(words_dbfilein, word_map_toidx, word_map_fromidx)
    conn = sqlite3.connect(words_dbfilein)
    w_in = conn.cursor()

    #concepts_conn = load_concepts_mem_db(concepts_dbfilein, concept_map_toidx)
    concepts_conn = sqlite3.connect(concepts_dbfilein)
    c_in = concepts_conn.cursor()

    word_idx = word_map_toidx[word]
    concept_idx = concept_map_toidx[concept]

    word_dict = {}
    w_in.execute('''select feature, pmi from words where word = ?''', (word_idx,))
    for feature, pmi in w_in:
        word_dict[feature] = pmi
    sim(word, c_in, word_dict, concept_idx)

def main():
    view_common('physics', 'agent.n.03')

if __name__ == "__main__":
    main()

