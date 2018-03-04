#!/usr/bin/python

# produces the file in the format
# ["word", [["concept", sim], ...]]
# using word-concept-topsim-no-parent-0.60-100

# the next step is to cluster the concept lists for each word to decide
# when multiple-inheritence should be used

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

words = set()
names = set()

fin = open('words-no-adverbs')
for line in fin:
	word = line.strip()
	words.add(word)
fin.close()

fin = open('proper_names')
for line in fin:
	name = line.strip()
	names.add(name)
fin.close()

def word_concept_sim(word, concept):
	synset = nlwn.synset(concept)
	is_similar = False
	max_sim = 0.0
	for s in nlwn.synsets(word, pos='n'):
		simval = s.lin_similarity(synset, features.ic)
		if simval > max_sim:
			max_sim = simval

	return max_sim

def is_too_sim(word, concept, t=0.30):

	for stem in nlwn._morphy(word, pos='n'):
		s = word_concept_sim(stem, concept)

		if s > t:
			return True

		synset = nlwn.synset(concept)
		for ss in nlwn.synsets(stem, pos='n'):
			if synset in features.all_hypernyms(ss):
				return True

		return False

def show_all(word_concept_topsim_filename):
	fin = open(word_concept_topsim_filename)
	prev_chosen = set()

	word_concept_dict = defaultdict(list)

	for line_num, line in enumerate(fin):
		if line_num in [0,1] and line.startswith('loaded'):
			continue

		word, concept_sim_list = json.loads(line)
		if word not in words:
			continue

		# throw out words that have instance hypernyms in WordNet
		has_instance_hyper = False
		has_upper = False
		for synset in nlwn.synsets(word, pos='n'):
			if synset.instance_hypernyms():
				has_instance_hyper = True
				break

			for lemma in synset.lemmas():
				if lemma.name()[0].isupper() and lemma.name()[1:].islower() and lemma.name()[1:].isalpha():
					has_upper = True



		if has_instance_hyper:
			continue

		if has_upper:
			continue



		concept, top_s = concept_sim_list[0]
		is_sim = is_too_sim(word, concept)

		if (word,concept) not in prev_chosen:
			if top_s > 0.067 and not is_sim:
				#if not is_sim:
				#	 print '*', word, ':', concept, '--', nlwn.synset(concept).definition
				#else:
				#	 print word, ':', concept, '--', nlwn.synset(concept).definition

				word_concept_dict[word].append((concept, top_s))
				prev_chosen.add((word,concept))

		for concept, top_s in concept_sim_list[1:]:
			if (word,concept) not in prev_chosen:
				is_sim = is_too_sim(word, concept)
				if top_s > 0.11 and not is_sim:
					#if not is_sim:
					#	 print '*', word, ':', concept, '--', nlwn.synset(concept).definition
					#else:
					#	 print word, ':', concept, '--', nlwn.synset(concept).definition

					prev_chosen.add((word,concept))
					word_concept_dict[word].append((concept, top_s))

	for word in word_concept_dict:
		concept_list = word_concept_dict[word]
		concept_list = filter_superconcept_dupes(concept_list)
		for concept, top_s in concept_list:
			print word, ':', concept, '--', nlwn.synset(concept).definition()

		#print json.dumps( (word, concept_list) )

def filter_superconcept_dupes(concept_list):
	concept_list.sort(key=operator.itemgetter(1), reverse=True)

	new_concept_list = []
	for concept, s in concept_list:
		synset = nlwn.synset(concept)
		super_exists = False
		for c2, s2 in new_concept_list:
			c2_synset = nlwn.synset(c2)
			if c2_synset in features.all_hypernyms(synset) or synset in features.all_hypernyms(c2_synset):
				super_exists = True
				break
		if not super_exists:
			new_concept_list.append( (concept, s) )
	return new_concept_list


def main():
	word_concept_topsim_filename = sys.argv[1]
	show_all(word_concept_topsim_filename)

if __name__ == "__main__":
	main()
