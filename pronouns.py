#!/usr/bin/python
from __future__ import with_statement
from nltk.corpus import wordnet as nlwn

import pdb

# singular pronouns refer to pronouns that come before the verb in simple
# non-question constructions like "<pronoun> <verb>", that change the form of
# the verb as singular noun subjects do
singular_pronouns = ['he', 'she', 'it', 'everyone', 'anyone', 'someone',\
        'anybody', 'anything', 'nobody', 'somebody', 'someone', 'something',\
        'that', 'this', 'what', 'whoever', 'whomever']

pronouns = ['all', 'another', 'any', 'anybody', 'anyone', 'anything', \
        'both', 'each', 'each other', 'either', 'everybody', 'everyone',\
        'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself',\
        'his', 'I', 'i', 'it', 'its', 'itself', 'little', 'many', 'me', 'mine', \
        'more', 'most', 'much', 'myself', 'neither', 'no one', 'nobody', \
        'none', 'nothing', 'one', 'one another', 'other', 'others', 'ours',\
        'ourselves', 'several', 'she', 'some', 'somebody', 'someone', 'something',\
        'that', 'theirs', 'them', 'themselves', 'these', 'they', 'this', 'those',\
        'us', 'we', 'what', 'whatever', 'which', 'whichever', 'who', 'whoever',\
        'whom', 'whomever', 'whose', 'you', 'yours', 'yourself', 'yourselves']

def is_ambiguous_pronoun(pronoun):
    if is_pronoun(pronoun):
        for unambig in pronoun_mapper.map_pronoun_to_synset:
            if unambig.lower() == pronoun.lower():
                return False
        return True
    return False

def is_singular_pronoun(pronoun):
    return pronoun.lower() in singular_pronouns

def is_pronoun(pronoun):
    return pronoun.lower() in pronouns

class pronoun_to_synset_mapper:
    def __init__(self):
        mapfile='pronouns.wnmap'
        self.map_pronoun_to_synset = {}
        self.map_pronoun_to_stem = {}
        with open(mapfile, 'r') as fin:
            for line in fin:
                pronoun, wn_word, synset_num = line.split()
                pronoun = ' '.join(pronoun.split('_'))
                self.map_pronoun_to_synset[pronoun] = nlwn.synsets(wn_word, pos=nlwn.NOUN)[int(synset_num)]
                self.map_pronoun_to_stem[pronoun] = wn_word

    def map(self, pronoun):
        if self.map_pronoun_to_synset.has_key(pronoun.lower()):
            return self.map_pronoun_to_synset[pronoun.lower()]

    def map_to_stem(self, pronoun):
        if self.map_pronoun_to_stem.has_key(pronoun.lower()):
            return self.map_pronoun_to_stem[pronoun.lower()]

# a utiltiy to map prononus to wn synsets
pronoun_mapper = pronoun_to_synset_mapper()

