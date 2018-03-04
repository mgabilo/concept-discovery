#!/usr/bin/env python
from __future__ import with_statement

#import nltk.wordnet as nlwn
from nltk.corpus import wordnet as nlwn
import pronouns as pn
import operator
import pdb
import heads as hd

# TODO: replace all synset.hypernyms() by this hypernym(synset) function
# same for hyponyms
def hypernyms(synset):
    return synset.hypernyms() + synset.instance_hypernyms()

def hyponyms(synset):
    return synset.hyponyms() + synset.instance_hyponyms()


containers = [nlwn.synset('container.n.01'), \
        nlwn.synset('helping.n.01'), \
        nlwn.synset('definite_quantity.n.01'), \
        nlwn.synset('indefinite_quantity.n.01'), \
        nlwn.synset('kind.n.01'),\
        nlwn.synset('degree.n.01')]

whole_number_synset = nlwn.synset('whole_number.n.01')
person_synset = nlwn.synset('person.n.01')

def lowest_common_hypernyms(synset_list):
    if len(synset_list) == 1:
        return [synset_list[0]]

    subsumers = []
    for i in xrange(len(synset_list)):
        for j in xrange(i+1, len(synset_list)):
            subsumers.extend( synset_list[i].lowest_common_hypernyms(synset_list[j]) )
    return [s for s in set(subsumers)]

def synsets_of_head(head):
    synsets = []
    for stem in head['STEMS']:
        synsets.extend(synsets_of_stem(stem))
    return synsets

def synsets_of_stems(stems):
    synsets = []
    for stem in stems:
        synsets.extend(synsets_of_stem(stem))
    return synsets

def synsets_of_stem(stem):
    if len(stem) == 0: return []

    synsets = nlwn.synsets(stem, pos=nlwn.NOUN)

    new_synsets = []
    for syn in synsets:
        hypernyms = syn.hypernyms()
        new_synsets.extend(hypernyms)

        for subhyp in syn.hypernyms():
            hypernyms = subhyp.hypernyms()
            new_synsets.extend(hypernyms)

        new_synsets.append(syn)

    return new_synsets


def synsets_expand(syn):

    new_synsets = []
    hypernyms = syn.hypernym_instances()
    new_synsets.extend(hypernyms)

    for subhyp in syn.hypernyms():
        hypernyms = subhyp.hypernym_instances()
        new_synsets.extend(hypernyms)

    new_synsets.append(syn)

    return new_synsets

def morphy2(noun, pos=nlwn.NOUN):
    noun = noun.replace(' ', '_').lower()
    return [n for n in nlwn._morphy(noun, pos) if len(nlwn.synsets(n, pos)) > 0 and len(n) > 0]

def morphy(word, pos=nlwn.NOUN):
    if pos == nlwn.VERB:
        word = word.lower()
        verb = ''
        for w in word.split():
            m = nlwn.morphy(w, pos=pos)
            if m:
                verb += m + ' '
            else:
                verb += w + ' '
        verb = verb.strip().replace(' ', '_')
        verb = nlwn.morphy(verb, pos)
        if verb:
            return verb

    return nlwn.morphy(word.replace(' ', '_'), pos)

# map the NE tags to wn synsets
map_ne_to_stem = {\
        'PER': 'person',\
        'ORG': 'organization',\
        'LOC': 'location',\
        'MISC': 'thing'}

map_ne_to_synset = {\
        'PER': nlwn.synset('person.n.01'),\
        'ORG': nlwn.synset('organization.n.01'),\
        'LOC': nlwn.synset('location.n.01'),\
        'MISC': nlwn.synset('thing.n.12')}

def noun_synsets(word):
    synsets = []
    for stem in morphy2(word):
        synsets.extend(nlwn.synsets(stem, pos=nlwn.NOUN))
    return synsets

def synset_subsumes(subsumer, synset):
    # does synset `subsumer' subsume synset `synset'?
    common_hyper = subsumer.common_hypernyms(synset)
    return subsumer in common_hyper

def synset_subsumes_word(subsumer, word):
    # is there is a synset of word such that subsumer subsumes it?
    subsume = [synset_subsumes(subsumer, synset) for synset in nlwn.synsets(word, pos=nlwn.NOUN)]
    return True in subsume

def sim_synsets(synsets1, synsets2):
    # returns the similarity between two lists of synsets.
    # this is the maximum similarity between any two synsets of synsets1 and synsets2
    # returns (simval, synset1, synset2)
    max_simval = -1
    max_tuple = None
    for s1 in synsets1:
        for s2 in synsets2:
            simval = s1.wup_similarity(s2)
            if simval > max_simval:
                max_simval = simval
                max_tuple = (max_simval, s1, s2)
    return max_tuple

def sim_synsets_stems(synsets, stems):
    # returns the similarity between synsets and all of the synsets of stems
    synsets2 = []
    for stem in stems: synsets2.extend(synsets_of_stem(stem))
    return sim_synsets(synsets, synsets2)

def sim_stems(stems1, stems2):
    synsets1 = []
    for stem in stems1: synsets1.extend(synsets_of_stem(stem))
    return sim_synsets_stems(synsets1, stems2)

def list_equal(l):
    # returns True if all items in list are equal. does extra useless compares.
    if len(l) <= 1: return True
    for a in l:
        for b in l:
            if a != b:
                return False
    return True

def nsd_similarity_pair(noun1_stems, noun2_stems, min_sim_thresh = 0.70):

    noun1_stems_mfs = [synsets_of_stem(n)[0] for n in noun1_stems]
    noun2_stems_mfs = [synsets_of_stem(n)[0] for n in noun2_stems]

    max_tuple = sim_synsets_stems(noun1_stems_mfs, noun2_stems)
    if max_tuple:
        simval, noun1_synset, noun2_synset = max_tuple
        if simval >= min_sim_thresh:
            return simval, noun1_synset, noun2_synset

    max_tuple = sim_synsets_stems(noun2_stems_mfs, noun1_stems)
    if max_tuple:
        simval, noun2_synset, noun1_synset = max_tuple
        if simval >= min_sim_thresh:
            return simval, noun1_synset, noun2_synset

    max_tuple = sim_stems(noun1_stems, noun2_stems)
    if max_tuple:
        simval, noun1_synset, noun2_synset = max_tuple
        print max_tuple
        if simval >= min_sim_thresh:
            return simval, noun1_synset, noun2_synset

#def synset_is_member_of(synset):
#    #e.g., synset_is_member_of(child_synset) returns people_synset
#    wholes = []
#    hypernyms = [s for s in synset.closure('hyp')] + [synset]
#    for s in hypernyms:
#        wholes.extend(s.member_meronyms())
#    return wholes

def nsd_similarity(nouns):

    sense_assign = [ [] for n in nouns ]

    if len(nouns) <= 1:
        return None

    n1_idx = 0
    while n1_idx < len(nouns):

        n2_idx = n1_idx + 1
        while n2_idx < len(nouns):
            n1 = nouns[n1_idx]
            n2 = nouns[n2_idx]
            senses = nsd_similarity_pair(n1, n2)
            if not senses:
                return None

            simval, synset1, synset2 = senses

            sense_assign[n1_idx].append(synset1)
            sense_assign[n2_idx].append(synset2)
            n2_idx += 1

        n1_idx += 1

    for s_list in sense_assign:
        if not list_equal(s_list):
            return None
     
    return [s[0] for s in sense_assign]

def has_wordnet_noun(phrase_list):
    for noun in phrase_list:
        if len(morphy2(noun.lower())) > 0:
            return noun

def count_content_words(string):
    count = 0
    for w in string.split():
        if is_content_word(w):
            count += 1
    return count

def is_content_word(word):
    word_lower = word.lower()
    apos = word_lower.find("'")
    if apos != -1:
        word_lower = word_lower[:apos]

    comma = word_lower.find(",")
    if comma != -1:
        word_lower = word_lower[:comma]

    if (pn.is_pronoun(word_lower) and not pn.is_ambiguous_pronoun(word_lower)) or word[0].isupper():
        return True

    if word_lower == '*' or word_lower in hd.ARTICLES:
        return False

    word_noun = morphy(word_lower, pos=nlwn.NOUN)
    if word_noun and len(nlwn.synsets(word_noun, pos=nlwn.NOUN)) > 0:
        return True

    word_adj = morphy(word_lower, pos=nlwn.ADJ)
    if word_adj and len(nlwn.synsets(word_adj, pos=nlwn.ADJ)) > 0:
        return True

    word_verb = morphy(word_lower, pos=nlwn.VERB)
    if word_verb and len(nlwn.synsets(word_verb, pos=nlwn.VERB)) > 0:
        return True

    word_adv = morphy(word, pos=nlwn.ADB)
    if word_adv and len(nlwn.synsets(word_adv, pos=nlwn.ADV)) > 0:
        return True

    return False

def stems_have_instance(stems):
    for stem in stems:
        for synset in nlwn.synsets(stem, pos=nlwn.NOUN):
            if len(synset.instance_hypernyms()) > 0:
                return True
    return False

def get_content_words_from_string(string):
    content_words = []
    string = string.replace('*', '')
    string = string.replace('"', '')
    string = string.replace(',', '')
    string = string.replace("'", '')
    for w in string.split():
        if is_content_word(w):
            content_words.append(w)
    return content_words
