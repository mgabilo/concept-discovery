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
feature_map_toidx, feature_map_fromidx = featuremap.load_feature_map()

def get_comparable_concepts(c_in, w_in, word_idx, word_dict):
    concept_set = set()
    word_features = set()
    for feature, pmi in word_dict.items():
        if pmi > 5.0:
            word_features.add(feature)

    w_in.execute('''select feature from words where word = ? and pmi > 5.0''', (word_idx,))
    for feature in w_in:
        feature = feature[0]
        if feature not in word_features:
            continue

        c_in.execute('''select concept from concepts where feature = ? and pmi > 5.0''', (feature,))
        for concept in c_in:
            concept = concept[0]
            if concept not in concept_set:
                concept_set.add(concept)

    return concept_set

def remove_intersection(c_in, word_dict, concept_idx):
    c_in.execute('''select feature, pmi from concepts where concept = ?''', (concept_idx,))
    for feature, pmi in c_in:
        if word_dict.has_key(feature):
            del word_dict[feature]

    return word_dict


def sim(sim_c, concept1_dict, concept2):
    sim_c.execute('''select feature, pmi from concepts where concept = ?''', (concept2,))
    concept2_dict = {}
    for feature, pmi in sim_c:
        feature_tup = feature_map_fromidx[feature]
        section, feature_name = feature_tup
        if section == 'modifiers_of_head':
            continue

        concept2_dict[feature] = pmi

    concept1_norm = 0
    for pmi in concept1_dict.values():
        concept1_norm += pmi*pmi
    concept1_norm = math.sqrt(concept1_norm)

    concept2_norm = 0
    for pmi in concept2_dict.values():
        concept2_norm += pmi*pmi
    concept2_norm = math.sqrt(concept2_norm)

    dp = 0
    for feature in concept2_dict:
        if concept1_dict.has_key(feature):
            dp += concept1_dict[feature] * concept2_dict[feature]
    return dp / (concept1_norm * concept2_norm)

def assign_committees_one_level(c_in, w_in, word_idx, word_dict):
    candidate_committees_list = []
    for concept_idx in get_comparable_concepts(c_in, w_in, word_idx, word_dict):
        s = sim(c_in, word_dict, concept_idx)
        candidate_committees_list.append( (concept_idx, s) )

    chosen_committees = []
    sorted_candidates = sorted(candidate_committees_list, key=operator.itemgetter(1), reverse=True)
    for max_committee, max_sim in sorted_candidates:
        if max_sim > ASSIGN_COMMITTEES_T:
            chosen_committees.append( (max_committee, max_sim) )

    #max_committee, max_sim = max(candidate_committees_list, key=operator.itemgetter(1))
    #if max_sim > ASSIGN_COMMITTEES_T:
    #    return [ (max_committee, max_sim) ]
    #return []

    return chosen_committees

def assign_committees(c_in, w_in, word, word_map_toidx, concept_map_toidx):
    committees_lil = []

    word_idx = word_map_toidx[word]

    word_dict = {}
    w_in.execute('''select feature, pmi from words where word = ?''', (word_idx,))
    for feature, pmi in w_in:
        feature_tup = feature_map_fromidx[feature]
        section, feature_name = feature_tup
        if section == 'modifiers_of_head':
            continue

        word_dict[feature] = pmi


    if len(word_dict.keys()) == 0:
        return []

    while word_dict_fcount(word_dict) > 40:
        committees_list = assign_committees_one_level(c_in, w_in, word_idx, word_dict)
        if len(committees_list) != 0:
            committees_lil.append( committees_list )
            concept_idx, s = committees_list[0]
        elif len(committees_list) == 0:
            break

        word_dict = remove_intersection(c_in, word_dict, concept_idx)
    return committees_lil

def word_dict_fcount(word_dict):
    high_fcount = 0
    for feature, pmi in word_dict.items():
        if pmi > 7.0:
            high_fcount += 1
    return high_fcount

def get_all_words(words_dbfilein):
    word_map_toidx, word_map_fromidx = featuremap.load_concept_map('words.idx')
    concept_map_toidx, concept_map_fromidx = featuremap.load_concept_map()

    conn = load_words_mem_db(words_dbfilein, word_map_toidx, word_map_fromidx)

def main():
    if len(sys.argv) != 3:
        print 'Arguments: <wordsdb> <conceptsdb>  \t(outputs to stdout)'
        return

    word_map_toidx, word_map_fromidx = featuremap.load_concept_map('words.idx')
    concept_map_toidx, concept_map_fromidx = featuremap.load_concept_map()

    words_dbfilein = sys.argv[1]
    concepts_dbfilein = sys.argv[2]

    conn = load_words_mem_db(words_dbfilein, word_map_toidx, word_map_fromidx)
    #conn = sqlite3.connect(words_dbfilein)
    w_in = conn.cursor()

    concepts_conn = load_concepts_mem_db(concepts_dbfilein, concept_map_toidx)
    #concepts_conn = sqlite3.connect(concepts_dbfilein)
    c_in = concepts_conn.cursor()

    fin = open('words-to-be-processed')
    for line in fin:
        word = line.strip()
        assign_committees_synset(c_in, w_in, word, word_map_toidx, concept_map_toidx, concept_map_fromidx)
    fin.close()

def assign_committees_synset(c_in, w_in, word, word_map_toidx, concept_map_toidx, concept_map_fromidx):
    committees_lil = assign_committees(c_in, w_in, word, word_map_toidx, concept_map_toidx)
    if len(committees_lil) > 0:
        for committee_list in committees_lil:

            chosen_committees = []
            for c,s in committee_list:
                c_name = concept_map_fromidx[c]
                synset = nlwn.synset(c_name)

                too_sim = False
                for chosen_committee, chosen_s in chosen_committees:
                    chosen_synset = nlwn.synset(chosen_committee)
                    wn_s = synset.lin_similarity(chosen_synset, features.ic)
                    if wn_s > 0.3:
                        too_sim = True
                        break
                if not too_sim:
                    chosen_committees.append((c_name, s))

            print simplejson.dumps((word,chosen_committees))

if __name__ == "__main__":
    main()

