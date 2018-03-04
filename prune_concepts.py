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
import featuremap

#concept_models = {}


def load_concept_model(concept, base_dir):
	try:
		model = features.PMINounModel(concept, base_dir)
		return model
	except IOError:
		return None

def prune_concepts(current_synset, base_dir, clusters_dir, new_dir, current_model=None):
	hyponym_list = []
	for hyponym in current_synset.hyponyms() + current_synset.instance_hyponyms():

		hypo_model = load_concept_model(hyponym.name(), base_dir)
		hyponym_list.append((hyponym, hypo_model))

		try:
			os.stat(hypo_model.lemma_to_filename(hyponym.name(), new_dir))
			hypo_model.load_from_pmi_file(new_dir, hyponym.name())
			print 'LOADING', hyponym.name()
		except OSError:
			prune_concepts(hyponym, base_dir, clusters_dir, new_dir, hypo_model)

	if current_model:
		print 'Current:', current_synset.name()

		union_max_list = []

		current_model_cluster = load_concept_model(current_model.noun, clusters_dir)
		if current_model_cluster:

			for hyponym, hypo_model in hyponym_list:

				# if hyponym.name() == 'person.n.01':
				# 	pdb.set_trace()

				if hypo_model:
					s = current_model_cluster.cosine_similarity_given_other_feature_dict(hypo_model.feature_dict())
					if s > 0.75 or hypo_model.high_fcount < 100 or len(hyponym.hyponyms()) == 0:
						union_max_list.append(hypo_model)
						if hypo_model.high_fcount < 100:
							print 'Union_max hyponym (few features):', hyponym.name(), s
						else:
							print 'Union_max hyponym:', hyponym.name(), s

					else:
						print 'Add hyponym:', hyponym.name(), s
						hypo_model.save_to_file(new_dir)

			for hypo_model in union_max_list:
				current_model.union_max(hypo_model)


def main():

	base_dir = sys.argv[1]
	clusters_dir = sys.argv[2]
	new_dir = sys.argv[3]

	prune_concepts(nlwn.synset('entity.n.01'), base_dir, clusters_dir, new_dir)

if __name__ == "__main__":
	main()
