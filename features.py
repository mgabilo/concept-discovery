#!/usr/bin/python

import pronouns as pn
import ucfparser as up
import wordnet as wn
import heads as hd
import saveparses as savep
import simplejson as json
import trees as tr
import operator
import pdb
import math
from nltk.corpus import wordnet as nlwn
import nltk.tree as nltr
import simplejson as json
from glob import glob
import sys
from bz2 import BZ2File
import traceback
import time
from math import log
import shutil
import os
from nltk.corpus.reader.wordnet import information_content
import nltk.corpus.reader.wordnet
import nltk
import nsddebug
import generalizepmi as gpmi

HIGH_SIM = 0.8
LOW_SIM =  0.3

HIGH_FCOUNT = 7.0

relatives_memo = {}

ic_reader = nltk.corpus.reader.wordnet.WordNetICCorpusReader(nltk.data.find('corpora/wordnet_ic'),'.*\.dat')
ic = ic_reader.ic('ic-bnc-resnik.dat')

nsd_debugger = nsddebug.nsd_debugger()

feature_num_to_names = {\
		0 : 'subject_head_of_verbs', \
		1 : 'object1_head_of_verbs' , \
		2 : 'object2_head_of_verbs', \
		3 : 'modifiers_of_head', \
		4 : 'modified_heads', \
		5 : 'prep_head_noun_of_pp_pairs', \
		6 : 'prep_head_noun_pairs', \
		7 : 'prep_verb_pairs', \
		8 : 'coheads'}

name_to_feature_num = {\
		'subject_head_of_verbs' : 0, \
		'object1_head_of_verbs' : 1 , \
		'object2_head_of_verbs' : 2, \
		'modifiers_of_head' : 3, \
		'modified_heads' : 4, \
		'prep_head_noun_of_pp_pairs' : 5, \
		'prep_head_noun_pairs' : 6, \
		'prep_verb_pairs' : 7, \
		'coheads' : 8}

context_names = ['verb_subject_contexts_freqs', \
		'verb_object1_contexts_freqs', \
		'verb_object2_contexts_freqs', \
		'modifier_contexts_freqs', \
		'modified_head_contexts_freqs', \
		'prep_head_noun_of_pp_pairs_contexts_freqs', \
		'prep_head_noun_pairs_contexts_freqs', \
		'prep_verb_pairs_contexts_freqs', \
		'head_noun_context_freqs']

context_name_to_feature_nums = {\
		context_names[0] : 0, \
		context_names[1] : 1, \
		context_names[2] : 2, \
		context_names[3] : 3, \
		context_names[4] : 4, \
		context_names[5] : 5, \
		context_names[6] : 6, \
		context_names[7] : 7, \
		context_names[8] : 8}






def map_feature_dict_to_names(feature_dict):
	new_feature_dict = {}
	for val in feature_num_to_names.values():
		new_feature_dict[val] = []

	for key in feature_dict:
		num,feature = key
		new_feature_dict[feature_num_to_names[num]].append((feature, feature_dict[key]))
	return new_feature_dict


def magnitude(vec):
	mag = 0
	for v in vec.values():
		mag += v*v
	return math.sqrt(mag)

def cosine_sim(vec1, vec2):
	s1 = set(vec1.keys())
	s2 = set(vec2.keys())
	common_features = s1.intersection(s2)
	dotproduct = 0.0
	for feature in common_features:
		dotproduct += vec1[feature] * vec2[feature]
		
		#nsd_debugger.write_row( ['%s(<b>%s</b>)' % (feature_num_to_names[feature[0]], feature[1]), vec1[feature]] )
	try:
		sim = dotproduct / (magnitude(vec1) * magnitude(vec2)) 
		return sim
	except ZeroDivisionError: return 0

def pmi(freq_joint, freq_x, freq_y, freq_all):
	return log(freq_joint, 2) - log(freq_x, 2) - log(freq_y, 2) + log(freq_all, 2)

class ContextSimilarity(object):
	def __init__(self, nounmodels):
		self.models = nounmodels
		self.context_dict = self.models.context_dict()
		
	def get_pmi_feature_dict(self, stem, feature_dict=None):
		if not feature_dict:
			stem_fd = self.models.models[stem].feature_dict()
		else:
			stem_fd = {}
			self.models.load_models(None, [stem])
			whole_stem_fd = self.models.models[stem].feature_dict()
			for k in feature_dict:
				if whole_stem_fd.has_key(k):
					stem_fd[k] = whole_stem_fd[k]
				else:
					stem_fd[k] = 1.0

		stem_freq = self.models.noun_freqs[stem]
		pmi_dict = {}

		for feature in stem_fd.keys():
			try:
				context_freq = self.context_dict[feature]
				joint_freq = stem_fd[feature]
				pmi_dict[feature] = pmi(joint_freq, stem_freq, context_freq, self.models.word_count)
			except KeyError, e:
				pass
				#print e
			except ValueError, e:
				pass
				#traceback.print_exc(file=sys.stdout)
				#print stem, stem_freq, joint_freq, context_freq
		return pmi_dict

	def similarity_stems(self, stem1, stem2):
		stem1_pmi = self.get_pmi_feature_dict(stem1)
		stem2_pmi = self.get_pmi_feature_dict(stem2)
		return cosine_sim(stem1_pmi, stem2_pmi)

def save_traceback(e):
	print 'EXCEPTION', e
	f = open('TRACEBACKS', 'a')
	traceback.print_exc(file=f)
	traceback.print_exc()
	f.close()


def modifiers(whole_np_pos_list, head_np_pos_list):
	if len(whole_np_pos_list)  == len(head_np_pos_list):
		return ([], [])
	mod = whole_np_pos_list[0:len(whole_np_pos_list) - len(head_np_pos_list)]

	# clean_mod words are probably modifiers of the head_np
	# dirty_mod words may or may not be modifiers, not recommended to treat them as modifiers
	clean_mod = []
	dirty_mod = []
	mod_list = clean_mod

	for m in reversed(mod):
		if m.node in ['CC', 'COMMA']:
			mod_list = dirty_mod
		if m.node not in ['POS', 'CC', 'COMMA', 'SYM', 'FW', 'PERIOD']:
			mod_list.insert(0, [m])
	return (clean_mod, dirty_mod)


def modifiers_and_head_nouns(tree):
	excluded_subtrees = []
	word_modifier_pairs = []

	for subtree in tree.subtrees():

		skip = False
		for excluded_tree in excluded_subtrees:
			if tr.has_subtree(excluded_tree, subtree):
				skip = True
		if skip:
			continue
		if subtree.node in ['NP', 'NP-COORD']:

			head_np_list = hd.head_np_of_np(subtree)
			for head_np in head_np_list:
				excluded_subtrees.append(head_np)

				candidates = hd.head_np_candidate_heads(head_np)
				if candidates:
					head = hd.choose_head_np_from_candidates(candidates)
					if head and len(head['STEMS']) > 0:

						clean_mod, dirty_mod = modifiers(tr.tree_pos(head_np), head['POS_LIST'])
						if len(clean_mod) > 0:
							word_modifier_pairs.append( (head['POS_LIST'], clean_mod) )
						else:
							word_modifier_pairs.append( (head['POS_LIST'], []) )
						for dm in dirty_mod:
							word_modifier_pairs.append( (dm, []) )

		if not isinstance(subtree[0], nltr.Tree):
			word_modifier_pairs.append( ([subtree], []) )

	return word_modifier_pairs

def pos_list_to_stems(pos_list):
	word = ' '.join([pos[0] for pos in pos_list])
	if len(pos_list) > 1:
		stems = wn.morphy2(word)
		if stems:
			return ('N', stems)

	if len(pos_list) == 1:
		pos_word = pos_list[0]
		
		if pos_word.node in ['JJ', 'JJS', 'JJR', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VB']:
			stems = wn.morphy2(word, pos=nlwn.ADJ)
			if stems:
				return ('ADJ', stems)

		if pos_word.node in ['CD']:
			if pos_word[0].startswith('$'):
				return ('N', ['monetary_unit'])
			try:
				if int(pos_word[0]) > 1200 and int(pos_word[0]) < 2200:
					return ('N', ['particular_date'])
			except ValueError:
				pass

			return ('N', ['cardinal_number'])

		if pos_word.node in ['NN', 'NNS', 'NNP', 'NNPS']:
			stems = wn.morphy2(pos_word[0], pos=nlwn.NOUN)
			if stems:
				return ('N', stems)

		if pos_word.node in ['PRP']:
			stem = pn.pronoun_mapper.map_to_stem( pos_word[0] )
			if stem:
				return ('N', [stem])

	return ('N', [])




def json_fix_load_dict(j):
	j = json.loads(j)
	for i in xrange(len(j)):
		j[i][0] = tuple(j[i][0])
	return tuple([tuple(l) for l in j])

def all_hypernyms(s):
	h = {}
	for slist in s.hypernym_paths():
		for s in slist:
			h[s] = True
	return h

class PMINounModel(object):
	def __init__(self, lemma=None, models_dir=None):
		self.noun = None
		self.sections = ['subject_head_of_verbs', 'object1_head_of_verbs', \
				'object2_head_of_verbs', 'modifiers_of_head', 'modified_heads', \
				'prep_head_noun_of_pp_pairs', 'prep_head_noun_pairs', \
				'prep_verb_pairs', 'coheads']
		for section in self.sections:
			self.__dict__[section] = {}
		if models_dir and lemma:
			self.load_from_pmi_file(models_dir, lemma)
		self.fcount = 0
		self.high_fcount = 0

	def load_from_pmi_file(self, models_dir, lemma):
		filename = self.lemma_to_filename(lemma, models_dir)
		self._load_from_pmi_file(filename)

	def feature_dict(self):
		d = {}
		for n, fdict_name in enumerate(self.sections):
			for k in self.__dict__[fdict_name].keys():
				d[ (n, k) ] = self.__dict__[fdict_name][k]
		return d

	def cosine_similarity_given_other_feature_dict(self, other_model_feature_dict):
		self_feature_dict = self.feature_dict()
		return cosine_sim(self_feature_dict, other_model_feature_dict)

	def lemma_to_filename(self, lemma, dir):
		return '%s/%s.pmi.bz2' % (dir, lemma.replace('/', '_'))

	def scale(self, weight):
		for section in self.sections:
			for k in self.__dict__[section]:
				self.__dict__[section][k] *= weight

	def save_to_file(self, dir):
		fout = BZ2File(self.lemma_to_filename(self.noun, dir), 'w')
		fout.write('%s\n' % self.noun)
		for section in self.sections:
			pmi_list = [(k, self.__dict__[section][k]) for k in self.__dict__[section]]
			pmi_list.sort(key=operator.itemgetter(1), reverse=True)
			save_pmi_list(pmi_list, fout, section)
		fout.close()

	def intersection(self, other_model, new_name=''):
		new_model = PMINounModel(new_name)
		for dname in self.sections:
			s1 = set(self.__dict__[dname].keys())
			s2 = set(other_model.__dict__[dname].keys())
			shared_features = s1.intersection(s2)
			for dkey in shared_features:
				new_model.__dict__[dname][dkey] = min(self.__dict__[dname][dkey], other_model.__dict__[dname][dkey])
		return new_model

	def union_max(self, pmi_noun_model):
		for section in pmi_noun_model.sections:

			for k in pmi_noun_model.__dict__[section]:
				tup_elem = None
				if isinstance(k, tuple):
					tup_elem, concept = k
				else:
					concept = k

				if not self.__dict__[section].has_key(k):
					self.__dict__[section][k] = pmi_noun_model.__dict__[section][k]
					if self.__dict__[section][k] > HIGH_FCOUNT:
						self.high_fcount += 1
					self.fcount += 1

				else:	 
					if self.__dict__[section][k] <= HIGH_FCOUNT and pmi_noun_model.__dict__[section][k] > HIGH_FCOUNT:
						self.high_fcount += 1

					self.__dict__[section][k] = max(pmi_noun_model.__dict__[section][k], self.__dict__[section][k])



	def _load_from_pmi_file(self, filename):
		fin = BZ2File(filename, 'r')
		self.noun = fin.readline().strip()
		self.fcount = 0
		self.high_fcount = 0
		section = None

		for section in self.sections:
			self.__dict__[section] = {}

		for line in fin:
			line = line.strip()
			
			if line.startswith('[') and line.endswith(']'):
				section = line[1:-1]
			else:
				if section == 'coheads' or section == 'modifiers_of_head':
					continue

				elem, eq, pmi = line.rpartition('=')
				elem_parts = tuple(elem.split())
				pmi = float(pmi)
				if pmi < 3.0: continue

				if len(elem_parts) > 2 or len(elem_parts) == 0:
					continue
				elif len(elem_parts) == 2:
					key = elem_parts
				elif len(elem_parts) == 1:
					key = elem_parts[0]


				if pmi > HIGH_FCOUNT:
					self.high_fcount += 1
				self.fcount += 1

				self.__dict__[section][key] = pmi
		fin.close()

def unambig_lemma_pairs(lemmas):
	for lemma_pair in pairs(lemmas):
		s1 = set(nlwn.synsets(lemma_pair[2], pos=nlwn.NOUN))
		s2 = set(nlwn.synsets(lemma_pair[3], pos=nlwn.NOUN))
		s3 = s1.intersection(s2)
		if len(s3) == 1:
			yield (lemma_pair[2], lemma_pair[3])

def noisy_concepts(concept, concept_lemma_synsets):
	noise = []
	for s in concept_lemma_synsets:
		if s != concept and s.wup_similarity(concept) < HIGH_SIM:
			noise.append(s)
	return noise


def synset_is_similar_to_any(synset, synset_list):
	for s in synset_list:
		if s.wup_similarity(synset) > LOW_SIM:
			return True
	return False

def hypernyms_by_level(child, level=2):
	levels = []
	levels.append([child])

	for l in xrange(1, level):
		if len(levels) == l:
			levels.append( [] )

		for prev_level_hyp in levels[l-1]:
			prev_hyp = prev_level_hyp.hypernyms()

			#similar_enough = False
			#for h in prev_hyp:
			#	 if h.wup_similarity(child) >= HIGH_SIM:
			#		 similar_enough = True
			#		 break
			#if not similar_enough:
			#	 break

			levels[l].extend(prev_hyp)

		if len(levels[l]) == 0:
			break

	return levels

def find_intersectable_parent_lemmas(child_synset):
	lemma_pairs = []
	#parent_lemma_weights = {}
	child_hypernym_levels = hypernyms_by_level(child_synset)

	for child_lemma in child_synset.lemmas:
		child_lemma_synsets = nlwn.synsets(child_lemma.name(), 'n')
		if len(child_lemma_synsets) <= 1: continue

		noise_synsets = noisy_concepts(child_synset, child_lemma_synsets)

		for child_hyp_level in child_hypernym_levels:
			for parent_synset in child_hyp_level:
				for parent_lemma in parent_synset.lemmas:
					if parent_lemma == child_lemma: continue
					#print 'parent synset and lemma', parent_lemma
					lemma_pair_is_noisy = False
					for parent_lemma_synset in nlwn.synsets(parent_lemma.name(), 'n'):
						#print '==>', parent_lemma_synset
						if parent_synset != parent_lemma_synset and synset_is_similar_to_any(parent_lemma_synset, noise_synsets):
							lemma_pair_is_noisy = True
					if not lemma_pair_is_noisy:
						lemma_pairs.append( (child_lemma.name(), parent_lemma.name()) )
						#parent_lemma_weights[parent_lemma.name] = remaining_ic / total_ic

				#remaining_ic = information_content(parent_synset, ic)

	return set([tuple(sorted(pair)) for pair in lemma_pairs])


def pairs(elem_list):
	for i in range(len(elem_list)):
		elem1 = elem_list[i]
		for offset, elem2 in enumerate(elem_list[i+1:]):
			j = offset + i + 1
			yield (i, j, elem1, elem2)

class PMISynsetModels(object):

	def __init__(self, noun_pmi_models_dir, synset_pmi_models_dir, dest_synset_dir = None):
		self.noun_pmi_models_dir = noun_pmi_models_dir
		self.synset_pmi_models_dir = synset_pmi_models_dir
		self.finished_models = {}
		self.finished_pr_models = {}
		self.dest_synset_dir = dest_synset_dir

		self.models = {}

	def load_models(self):
		print 'Loading SPMI files'
		for filename in glob('%s/*.spmi' % (self.synset_pmi_models_dir)):
			model = PMISynsetModel(None, self.noun_pmi_models_dir, self.synset_pmi_models_dir)
			model.load_from_synset_pmi_file(filename)
			self.models[model.synset_name] = model
		print 'done'

	def create_mr_pmi_synset_models(self, root_synset=nlwn.synsets('entity', 'n')[0]):
		
		# we have
		# 1. current_pmi_synset_model, that's our monosemous relatives from phase I
		# 2. hyponym_models, that's the "fully disambiguated" hyponyms

		if self.finished_models.has_key(root_synset.name):
			pmi_synset_model = PMISynsetModel(root_synset.name, self.noun_pmi_models_dir, self.synset_pmi_models_dir)
			pmi_synset_model.load_from_synset_pmi_file()
			return pmi_synset_model

		hyponym_models = []
		for hyponym in root_synset.hyponyms():
			hyponym_models.append(self.create_mr_pmi_synset_models(hyponym))

		#print '\nAt', root_synset.name
		#for h in hyponym_models:
		#	 print '==>', h.synset_name

		pmi_synset_model = PMISynsetModel(root_synset.name, self.noun_pmi_models_dir, self.synset_pmi_models_dir)
		pmi_synset_model.merge_hyponyms_and_monosemous_lemmas(hyponym_models)
		pmi_synset_model.save_to_file()
		self.finished_models[pmi_synset_model.synset_name] = True
		return pmi_synset_model

	def create_pr_pmi_synset_models(self, root_synset=nlwn.synsets('entity', 'n')[0]):
		
		current_pmi_synset_model = PMISynsetModel(root_synset.name, self.noun_pmi_models_dir, self.synset_pmi_models_dir, self.dest_synset_dir)
		current_pmi_synset_model.load_from_synset_pmi_file()

		if self.finished_pr_models.has_key(root_synset.name):
			return current_pmi_synset_model

		hyponym_models = []
		for hyponym in root_synset.hyponyms():
			hyponym_models.append(self.create_pr_pmi_synset_models(hyponym))

		intersected_models = []

		self.finished_pr_models[current_pmi_synset_model.synset_name] = True
		child_synset = nlwn.synset(current_pmi_synset_model.synset_name)
		for l1, l2 in find_intersectable_parent_lemmas(child_synset):
			if l1 in current_pmi_synset_model.merged_models and l2 in current_pmi_synset_model.merged_models:
				continue

			try:
				l1_noun_model = PMINounModel(l1.lower(), self.noun_pmi_models_dir)
				l2_noun_model = PMINounModel(l2.lower(), self.noun_pmi_models_dir)
				intersected_models.append(l1_noun_model.intersection(l2_noun_model))
				if l1 not in current_pmi_synset_model.merged_models:
					current_pmi_synset_model.merged_models.append(l1)
				if l2 not in current_pmi_synset_model.merged_models:
					current_pmi_synset_model.merged_models.append(l2)
			except IOError: pass

		for model in hyponym_models:
			w = current_pmi_synset_model.weight(model)
			current_pmi_synset_model.union_max(model, w)

		for model in intersected_models:
			current_pmi_synset_model.union_max(model, 1.0)

		current_pmi_synset_model.save_to_file()

		return current_pmi_synset_model




class PMISynsetModel(object):
	def __init__(self, synset_name, noun_pmi_models_dir, synset_pmi_models_dir, dest_synset_dir = None):

		self.merged_models = [] # LEMMAS
		self.synset_name = synset_name
		self.sections = ['subject_head_of_verbs', 'object1_head_of_verbs', \
				'object2_head_of_verbs', 'modifiers_of_head', 'modified_heads', \
				'prep_head_noun_of_pp_pairs', 'prep_head_noun_pairs', \
				'prep_verb_pairs', 'coheads']
		for section in self.sections:
			self.__dict__[section] = {}
		self.noun_pmi_models_dir = noun_pmi_models_dir
		self.synset_pmi_models_dir = synset_pmi_models_dir
		self.dest_synset_dir = dest_synset_dir
		self.ic = information_content(nlwn.synset(self.synset_name), ic)

	def cosine_similarity(self, other_model):
		self_feature_dict = self.feature_dict()
		other_feature_dict = other_model.feature_dict()

		return cosine_sim(self_feature_dict, other_feature_dict)

	def cosine_similarity_given_other_feature_dict(self, other_model_feature_dict):
		print 'cosine sim %s' % self.synset_name
		self_feature_dict = self.feature_dict()
		return cosine_sim(self_feature_dict, other_model_feature_dict)
	
	def feature_dict(self):
		d = {}
		for n, fdict_name in enumerate(self.sections):
			for k in self.__dict__[fdict_name].keys():
				d[ (n, k) ] = self.__dict__[fdict_name][k]
		return d

	def weight(self, hypo_model):
		if hypo_model.ic == 0 or self.ic == 0:
			weight = 1.0
		else:
			weight = self.ic / hypo_model.ic
		return weight

	# input: loaded synset models for all of our direct hyponyms
	# effect: we average_add the hyponyms into us and ...
	# look at each of our lemmas:
	#	if the lemma is monosemous and the lemma is not in self.merged_models,
	#	load it up from the file, and average_add it
	def merge_hyponyms_and_monosemous_lemmas(self, hyponym_models):
		for hypo in hyponym_models:
			w = self.weight(hypo)
			self.union_max(hypo, w)
			self.merged_models.extend(hypo.merged_models)
			#print '==> merging from hyponym', hypo.synset_name
		try:
			for lemma in nlwn.synset(self.synset_name).lemmas:
				lemma = lemma.name().lower()
				try:
					if lemma not in self.merged_models and len(nlwn.synsets(lemma, 'n')) == 1:
						try:
							pmi_noun_model = PMINounModel(lemma, self.noun_pmi_models_dir)
							self.union_max(pmi_noun_model, 1.0)
							self.merged_models.append(lemma)
							#print '==> merging from lemma pmi file', lemma
						except IOError, e:
							pass
				except ValueError, e:
					save_traceback(e)
		except ValueError, e:
			save_traceback(e)

	def save_to_file(self):
		# write the header:
		# synset_name
		# lemma1 lemma2 lemma3 ...

		if self.dest_synset_dir:
			save_dir = self.dest_synset_dir
		else:
			save_dir = self.synset_pmi_models_dir

		fout = open('%s/%s.spmi' % (save_dir, self.synset_name.lower().replace('/', '_')), 'w')
		fout.write('%s\n' % self.synset_name)
		for m in self.merged_models:
			fout.write('%s ' % m)
		fout.write('\n')

		for section in self.sections:
			fout.write('[%s]\n' % section)

			sorted_section = sorted([(k, self.__dict__[section][k]) for k in self.__dict__[section]], key=operator.itemgetter(1), reverse=True)
			for k, v in sorted_section:
				if isinstance(k, tuple):
					fout.write('%s %s=%s\n' % (k[0], k[1], v))
				else:
					fout.write('%s=%s\n' % (k, v))
		fout.close()


	def load_from_synset_pmi_file(self, filename=None):
		if not filename:
			fin = open('%s/%s.spmi' % (self.synset_pmi_models_dir, self.synset_name.lower().replace('/', '_')), 'r')
		else:
			fin = open(filename)

		self.synset_name = fin.readline().strip()
		for lemma in fin.readline().split():
			self.merged_models.append(lemma)

		section = None
		for line in fin:
			line = line.strip()
			
			if line.startswith('[') and line.endswith(']'):
				section = line[1:-1]
			else:
				elem, eq, pmi = line.rpartition('=')
				elem_parts = tuple(elem.split())
				pmi = float(pmi)

				if len(elem_parts) > 2 or len(elem_parts) == 0:
					print '???:', line

				elif len(elem_parts) == 2:
					key = elem_parts
				elif len(elem_parts) == 1:
					key = elem_parts[0]
				self.__dict__[section][key] = pmi
		fin.close()

	def union_max(self, pmi_noun_model, weight = 1.0):
		for section in pmi_noun_model.sections:

			for k in pmi_noun_model.__dict__[section]:
				if not self.__dict__[section].has_key(k):
					self.__dict__[section][k] = pmi_noun_model.__dict__[section][k] * weight
				else:	 
					self.__dict__[section][k] = max(pmi_noun_model.__dict__[section][k], self.__dict__[section][k] * weight)


def make_monosemous_relatives(wc_dir, synset_dir):
	try:
		os.mkdir(synset_dir)
	except OSError: pass
	models = PMISynsetModels(wc_dir, synset_dir)
	models.create_mr_pmi_synset_models()

def make_polysemous_relatives(wc_dir, synset_dir, dest_synset_dir):
	try:
		os.mkdir(dest_synset_dir)
	except OSError: pass
	models = PMISynsetModels(wc_dir, synset_dir, dest_synset_dir)
	models.create_pr_pmi_synset_models()

def save_pmi_list(pmi_list, fout, feature_name):
	fout.write('[%s]\n' % feature_name)
	for name, pmi in pmi_list:
		#if float(pmi) < 0.01: continue
		if isinstance(name, tuple):
			fout.write('%s %s=%s\n' % (name[0], name[1], pmi))
		else:
			fout.write('%s=%s\n' % (name, pmi))

def calculate_and_save_pmi(noun, context_sim, wc_dir):
	pmi_dict = context_sim.get_pmi_feature_dict(noun)
	pmi_names_dict = {}
	for num, feature in pmi_dict:
		feature_name = feature_num_to_names[num]

		if not pmi_names_dict.has_key(feature_name):
			pmi_names_dict[feature_name] = {}
		pmi_names_dict[feature_name][feature] = pmi_dict[ (num, feature) ]

	fout = BZ2File('%s/%s.pmi.bz2' % (wc_dir, noun.replace('/', '_')), 'w')
	fout.write('%s\n' % noun)
	for pmi_d_name in pmi_names_dict:
		pmi_d = pmi_names_dict[pmi_d_name]
		pmi_list = sorted([(name, pmi_d[name]) for name in pmi_d], key=operator.itemgetter(1), reverse=True)
		save_pmi_list(pmi_list, fout, pmi_d_name)
	fout.close()

class NounModel(object):

	#def pretty_print(self):
	#	 fdict = self.feature_dict()
	#	 for f in fdict:
	#		 print f, fdict[f]


	def calculate_and_save_pmi(self, noun_models, context_sim, wc_dir):
		# global function
		calculate_and_save_pmi(self.noun, context_sim, wc_dir)

	def save_to_file(self, WC_DIR):
		fout = open('%s/%s.freq' % (WC_DIR, self.noun.replace('/', '_')), 'w')
		fout.write(self.save_as_string())
		fout.close()


	def feature_dict(self):
		feature_list = [self.subject_head_of_verbs, self.object1_head_of_verbs, self.object2_head_of_verbs, \
				self.modifiers_of_head, self.modified_heads, self.prep_head_noun_of_pp_pairs, \
				self.prep_head_noun_pairs, self.prep_verb_pairs, self.coheads]

		d = {}
		for n, fdict in enumerate(feature_list):
			for k in fdict.keys():
				d[ (n, k) ] = fdict[k]
		return d

	def merge(self, other_model):
		for dname in self.feature_dict_names:
			for k in other_model.__dict__[dname]:
				if not self.__dict__[dname].has_key(k):
					self.__dict__[dname][k] = 0
				self.__dict__[dname][k] += other_model.__dict__[dname][k]



	def __init__(self, noun=''):
		self.noun = noun

		self.feature_dict_names = ['subject_head_of_verbs', 'object1_head_of_verbs', \
				'object2_head_of_verbs', 'modifiers_of_head', 'modified_heads', \
				'prep_head_noun_of_pp_pairs', 'prep_head_noun_pairs', \
				'prep_verb_pairs', 'coheads']
		self.sections = self.feature_dict_names

		# when noun acts as the head of a subject, then subject_head_of_verbs
		# are those verbs -- lowercase and lemmatized
		#
		# what verbs tend to select for noun?
		self.subject_head_of_verbs = {}
		self.object1_head_of_verbs = {}
		self.object2_head_of_verbs = {}

		# when noun acts as a head of an NP, then modifiers_of_head are the
		# words that come before it within the NP -- lowercase and lemmatized 
		#
		# what modifiers tend to select for noun?
		self.modifiers_of_head = {}

		self.modified_heads = {}

		# when noun acts as a head, these are the
		# (preposition heading the PP, head noun of the NP embedded within the PP)
		# pairs, when a PP may be attached to the noun (often times it is
		# attached to the verb, and it is added anyways -- but if a noun truly
		# selects for a preposition, there should be a good sign in here)
		#
		# what prepositions and head nouns of those PPs does noun tend to select for?
		self.prep_head_noun_of_pp_pairs = {}

		# when noun is head of a PP, these are the (prep, noun_head) pairs
		# where noun_head is what the PP attaches to
		self.prep_head_noun_pairs = {}

		# when noun is head of a PP, these are the (prep, verb) pairs
		# where verb is what the PP attaches to
		self.prep_verb_pairs = {}

		self.coheads = {}

		self.coheads_synsets = {}
		self.prep_head_noun_pairs_synsets = {}
		self.prep_head_noun_of_pp_pairs_synsets = {}
		self.modified_heads_synsets = {}

		self.coheads_scores = []
		self.prep_noun_scores = []
		self.prep_noun_pp_scores = []
		self.modified_heads_scores = []

		self.last_wc_dir = None

	def save_as_string(self):
		subject = json.dumps(tuple([(k, self.subject_head_of_verbs[k]) for k in self.subject_head_of_verbs]))
		object1 = json.dumps(tuple([(k, self.object1_head_of_verbs[k]) for k in self.object1_head_of_verbs]))
		object2 = json.dumps(tuple([(k, self.object2_head_of_verbs[k]) for k in self.object2_head_of_verbs]))
		modifiers = json.dumps(tuple([(k, self.modifiers_of_head[k]) for k in self.modifiers_of_head]))
		modifieds = json.dumps(tuple([(k, self.modified_heads[k]) for k in self.modified_heads]))
		pp = json.dumps(tuple([(k, self.prep_head_noun_pairs[k]) for k in self.prep_head_noun_pairs]))
		pp2 = json.dumps(tuple([(k, self.prep_head_noun_of_pp_pairs[k]) for k in self.prep_head_noun_of_pp_pairs]))
		pp3 = json.dumps(tuple([(k, self.prep_verb_pairs[k]) for k in self.prep_verb_pairs]))
		coheads = json.dumps(tuple([(k, self.coheads[k]) for k in self.coheads]))
		return json.dumps((self.noun, subject, object1, object2, modifiers, modifieds, pp, pp2, pp3, coheads))


	def load_from_string(self, json_str):
		self.noun, subject, object1, object2, modifiers, modifieds, pp, pp2, pp3, coheads = json.loads(json_str)
		subject = tuple([tuple(l) for l in json.loads(subject)])
		object1 = tuple([tuple(l) for l in json.loads(object1)])
		object2 = tuple([tuple(l) for l in json.loads(object2)])
		modifieds = tuple([tuple(l) for l in json.loads(modifieds)])
		coheads = tuple([tuple(l) for l in json.loads(coheads)])

		pp = json_fix_load_dict(pp)
		pp2 = json_fix_load_dict(pp2)
		pp3 = json_fix_load_dict(pp3)
		modifiers = json_fix_load_dict(modifiers)

		self.subject_head_of_verbs = dict(subject)
		self.object1_head_of_verbs = dict(object1)
		self.object2_head_of_verbs = dict(object2)
		self.modifiers_of_head = dict(modifiers)
		self.modified_heads = dict(modifieds)
		self.prep_head_noun_pairs = dict(pp)
		self.prep_head_noun_of_pp_pairs = dict(pp2)
		self.prep_verb_pairs = dict(pp3)
		self.coheads = dict(coheads)

	def pretty_print(self, min=0):
		print 'MODIFIERS of << %s >> when it acts as a head of an NP' % self.noun
		print '%-20s%s' % ('Modifier','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.modifiers_of_head[k]) for k in self.modifiers_of_head], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.0f' % (k,v)

		print 'HEADS MODIFIED by << %s >> ' % self.noun
		print '%-20s%s' % ('Head','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.modified_heads[k]) for k in self.modified_heads], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.0f' % (k,v)

		print 'CO-HEADS of << %s >>' % self.noun
		print '%-20s%s' % ('Co-head','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.coheads[k]) for k in self.coheads], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.0f' % (k,v)

		print '\nVERBS of scopes in which << %s >> acts as the SUBJECT\'s head noun' % self.noun
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.subject_head_of_verbs[k]) for k in self.subject_head_of_verbs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.1f' % (k,v)

		print '\nVERBS of scopes in which << %s >> acts as the OBJECT-1\'s head noun' % self.noun
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.object1_head_of_verbs[k]) for k in self.object1_head_of_verbs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.1f' % (k,v)

		print '\nVERBS of scopes in which << %s >> acts as the OBJECT-2\'s head noun' % self.noun
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.object2_head_of_verbs[k]) for k in self.object2_head_of_verbs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%.1f' % (k,v)

		print '\n(PREP, HEAD-NOUN-OF-NP-EMBEDDED-WITHIN-PP) pairs which possibly attach to << %s >>' % self.noun
		print '%-20s%-20s%s' % ('Preposition', 'Head PP', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_head_noun_of_pp_pairs[k]) for k in self.prep_head_noun_of_pp_pairs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%-20s%.1f' % (k[0], k[1], v)

		print '\n(PREP, HEAD-NOUN) pairs to which PP headed by << %s >> attaches' % self.noun
		print '%-20s%-20s%s' % ('Preposition', 'Head Noun of NP', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_head_noun_pairs[k]) for k in self.prep_head_noun_pairs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%-20s%.1f' % (k[0], k[1], v)

		print '\n(PREP, VERB) pairs to which PP headed by << %s >> attaches' % self.noun
		print '%-20s%-20s%s' % ('Preposition', 'Head Noun of NP', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_verb_pairs[k]) for k in self.prep_verb_pairs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min:
				print '%-20s%-20s%.1f' % (k[0], k[1], v)

	def add_subject_head_of_verb(self, verb, credit):

		if not self.subject_head_of_verbs.has_key(verb):
			self.subject_head_of_verbs[verb] = 0
		self.subject_head_of_verbs[verb] += credit

	def add_object1_head_of_verb(self, verb, credit):

		if not self.object1_head_of_verbs.has_key(verb):
			self.object1_head_of_verbs[verb] = 0
		self.object1_head_of_verbs[verb] += credit

	def add_object2_head_of_verb(self, verb, credit):

		if not self.object2_head_of_verbs.has_key(verb):
			self.object2_head_of_verbs[verb] = 0
		self.object2_head_of_verbs[verb] += credit

	def add_modifier_of_head(self, modifier, credit):

		if not self.modifiers_of_head.has_key(modifier):
			self.modifiers_of_head[modifier] = 0
		self.modifiers_of_head[modifier] += credit

	def add_modified_head(self, head, credit):

		if not self.modified_heads.has_key(head):
			self.modified_heads[head] = 0
		self.modified_heads[head] += credit

	def add_cohead(self, cohead, credit):

		if not self.coheads.has_key(cohead):
			self.coheads[cohead] = 0
		self.coheads[cohead] += credit


	def add_prep_head_noun_of_pp_pair(self, prep, head_pp, credit):

		if not self.prep_head_noun_of_pp_pairs.has_key((prep, head_pp)):
			self.prep_head_noun_of_pp_pairs[(prep, head_pp)] = 0
		self.prep_head_noun_of_pp_pairs[(prep, head_pp)] += credit

	def add_prep_head_noun_pair(self, prep, head_noun, credit):

		if not self.prep_head_noun_pairs.has_key((prep, head_noun)):
			self.prep_head_noun_pairs[(prep, head_noun)] = 0
		self.prep_head_noun_pairs[(prep, head_noun)] += credit

	def add_prep_verb_pair(self, prep, verb, credit):

		if not self.prep_verb_pairs.has_key((prep, verb)):
			self.prep_verb_pairs[(prep, verb)] = 0
		self.prep_verb_pairs[(prep, verb)] += credit


class NounModels(object):


	def load_scores(self, wc_dir):
		for model in self.models.values():
			model.load_scores(wc_dir)

	def calculate_and_save_pmi(self, wc_dir):
		context_sim = ContextSimilarity(self)
		for model in self.models.values():
			try:
				model.calculate_and_save_pmi(self, context_sim, wc_dir)
			except KeyError,e:
				pass
				#traceback.print_exc(file=sys.stdout)
				#print 'KeyError', e, 'trying to save pmi for', model.noun

	def calculate_and_save_synset_scores(self, wc_dir):
		for model in self.models.values():
			try:
				model.calculate_and_save_synset_scores(self.ic, self, wc_dir)
			except KeyError,e:
				traceback.print_exc(file=sys.stdout)
				print 'KeyError', e, 'trying to save score for', model.noun

	def context_dict(self):
		context_list = [self.verb_subject_contexts_freqs, self.verb_object1_contexts_freqs, self.verb_object1_contexts_freqs, \
				self.modifier_contexts_freqs, self.modified_head_contexts_freqs, self.prep_head_noun_of_pp_pairs_contexts_freqs, \
				self.prep_head_noun_pairs_contexts_freqs, self.prep_verb_pairs_contexts_freqs, \
				self.head_noun_context_freqs]


		d = {}
		for n, cdict in enumerate(context_list):
			for k in cdict.keys():
				d[ (n, k) ] = cdict[k]
		return d

	def merge(self, other_nounmodels):
		for dname in self.context_dict_names:
			for k in other_nounmodels.__dict__[dname]:
				if not self.__dict__[dname].has_key(k):
					self.__dict__[dname][k] = 0
				self.__dict__[dname][k] += other_nounmodels.__dict__[dname][k]
		self.all_noun_context_count += other_nounmodels.all_noun_context_count
		self.word_count += other_nounmodels.word_count

		for other_model_key in other_nounmodels.models:
			if not self.models.has_key(other_model_key):
				self._init_models_with_key(other_model_key)
			self.models[other_model_key].merge( other_nounmodels.models[other_model_key] )

	def __init__(self):

		self.totals_loaded = False

		self.ic_reader = nltk.corpus.reader.wordnet.WordNetICCorpusReader(nltk.data.find('corpora/wordnet_ic'),'.*\.dat')
		self.ic = self.ic_reader.ic('ic-bnc-resnik.dat')

		self.models = {}
		self.context_dict_names = ['verb_object1_contexts_freqs', 'verb_object2_contexts_freqs', \
				'verb_subject_contexts_freqs', 'prep_head_noun_of_pp_pairs_contexts_freqs', \
				'prep_head_noun_pairs_contexts_freqs', 'prep_verb_pairs_contexts_freqs', \
				'modifier_contexts_freqs', 'modified_head_contexts_freqs', \
				'head_noun_context_freqs', 'noun_freqs']

		self.verb_object1_contexts_freqs = {}
		self.verb_object2_contexts_freqs = {}
		self.verb_subject_contexts_freqs = {}
		self.prep_head_noun_of_pp_pairs_contexts_freqs = {}
		self.prep_head_noun_pairs_contexts_freqs = {}
		self.prep_verb_pairs_contexts_freqs = {}
		self.modifier_contexts_freqs = {}
		self.modified_head_contexts_freqs = {}
		self.head_noun_context_freqs = {}

		self.noun_freqs = {}
		self.all_noun_context_count = 0
		self.word_count = 0

	def pretty_print(self, min=0):
		print 'VERB subject	 Contexts'
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.verb_subject_contexts_freqs[k]) for k in self.verb_subject_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print 'VERB object-1  Contexts'
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.verb_object1_contexts_freqs[k]) for k in self.verb_object1_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print 'VERB object-2  Contexts'
		print '%-20s%s' % ('Verb','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.verb_object2_contexts_freqs[k]) for k in self.verb_object2_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print '\n(PREP, HEAD-NOUN-OF-NP-EMBEDDED-WITHIN-PP) pairs Contexts (attach any noun)'
		print '%-20s%-20s%s' % ('Preposition', 'Head Noun of NP', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_head_noun_of_pp_pairs_contexts_freqs[k]) for k in self.prep_head_noun_of_pp_pairs_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%-20s%.1f' % (k[0], k[1], v)

		print '\n(PREP, HEAD-NOUN) pairs Contexts (any noun attaches)'
		print '%-20s%-20s%s' % ('Preposition', 'Head Noun of NP', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_head_noun_pairs_contexts_freqs[k]) for k in self.prep_head_noun_pairs_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%-20s%.1f' % (k[0], k[1], v)

		print '\n(PREP, VERB) pairs Contexts (any noun attaches)'
		print '%-20s%-20s%s' % ('Preposition', 'Verb', 'Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.prep_verb_pairs_contexts_freqs[k]) for k in self.prep_verb_pairs_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%-20s%.1f' % (k[0], k[1], v)

		print '\nMODIFIER  Contexts'
		print '%-20s%s' % ('Modifier','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.modifier_contexts_freqs[k]) for k in self.modifier_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print '\nMODIFIED HEADS Contexts'
		print '%-20s%s' % ('Head','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.modified_head_contexts_freqs[k]) for k in self.modified_head_contexts_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print '\nHEAD NOUN Contexts'
		print '%-20s%s' % ('Head','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.head_noun_context_freqs[k]) for k in self.head_noun_context_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)

		print '\nNOUN COUNT %d' % self.all_noun_context_count
		print '%-20s%s' % ('Word','Frequency')
		print '-----------------------------------------------------'
		m = sorted([(k, self.noun_freqs[k]) for k in self.noun_freqs], key=operator.itemgetter(1), reverse=True)
		for k,v in m:
			if v > min: print '%-20s%.0f' % (k,v)
		print '\nWORD COUNT = %d' % self.word_count

	def save_as_string(self):
		o1 = json.dumps(tuple([(k, self.verb_object1_contexts_freqs[k]) for k in self.verb_object1_contexts_freqs]))
		o2 = json.dumps(tuple([(k, self.verb_object2_contexts_freqs[k]) for k in self.verb_object2_contexts_freqs]))
		s = json.dumps(tuple([(k, self.verb_subject_contexts_freqs[k]) for k in self.verb_subject_contexts_freqs]))
		pp = json.dumps(tuple([(k, self.prep_head_noun_of_pp_pairs_contexts_freqs[k]) for k in self.prep_head_noun_of_pp_pairs_contexts_freqs]))
		pp2 = json.dumps(tuple([(k, self.prep_head_noun_pairs_contexts_freqs[k]) for k in self.prep_head_noun_pairs_contexts_freqs]))
		pp3 = json.dumps(tuple([(k, self.prep_verb_pairs_contexts_freqs[k]) for k in self.prep_verb_pairs_contexts_freqs]))
		mod = json.dumps(tuple([(k, self.modifier_contexts_freqs[k]) for k in self.modifier_contexts_freqs]))
		mod2 = json.dumps(tuple([(k, self.modified_head_contexts_freqs[k]) for k in self.modified_head_contexts_freqs]))
		h = json.dumps(tuple([(k, self.head_noun_context_freqs[k]) for k in self.head_noun_context_freqs]))
		noun = json.dumps(tuple([(k, self.noun_freqs[k]) for k in self.noun_freqs]))

		return json.dumps((o1, o2, s, pp, pp2, pp3, mod, mod2, h, noun, self.all_noun_context_count, self.word_count))

	def load_from_string(self, json_str):
		o1, o2, s, pp, pp2, pp3, mod, mod2, h, noun, self.all_noun_context_count, self.word_count = json.loads(json_str)
		o1 = tuple([tuple(l) for l in json.loads(o1)])
		o2 = tuple([tuple(l) for l in json.loads(o2)])
		s = tuple([tuple(l) for l in json.loads(s)])
		mod2 = tuple([tuple(l) for l in json.loads(mod2)])
		h = tuple([tuple(l) for l in json.loads(h)])
		noun = tuple([tuple(l) for l in json.loads(noun)])
		mod = json_fix_load_dict(mod)
		pp = json_fix_load_dict(pp)
		pp2 = json_fix_load_dict(pp2)
		pp3 = json_fix_load_dict(pp3)

		self.verb_object1_contexts_freqs = dict(o1)
		self.verb_object2_contexts_freqs = dict(o2)
		self.verb_subject_contexts_freqs = dict(s)
		self.prep_head_noun_of_pp_pairs_contexts_freqs = dict(pp)
		self.prep_head_noun_pairs_contexts_freqs = dict(pp2)
		self.prep_verb_pairs_contexts_freqs = dict(pp3)
		self.modifier_contexts_freqs = dict(mod)
		self.modified_head_contexts_freqs = dict(mod2)
		self.head_noun_context_freqs = dict(h)
		self.noun_freqs = dict(noun)

	def save_models_to_files(self, WC_DIR):
		print 'Saving models to files...', WC_DIR
		try:
			os.mkdir(WC_DIR)
		except OSError: pass

		models_totals_str = self.save_as_string()
		fout = open('%s/noun_models.totals' % (WC_DIR), 'w')
		fout.write('%s\n' % models_totals_str)
		fout.close()

		for m in self.models.values():
			m.save_to_file(WC_DIR)
		print 'Done saving models.'

	def load_models(self, WC_DIR, models_to_load=None):
		if WC_DIR:
			self.last_wc_dir = WC_DIR
		if not WC_DIR:
			WC_DIR = self.last_wc_dir

		if not self.totals_loaded:
			print 'Loading frequency models from directory', WC_DIR
			models_fin = BZ2File('%s/noun_models.totals.bz2' % WC_DIR, 'r')
			self.load_from_string( models_fin.readline() )
			models_fin.close()
			self.totals_loaded = True

		if models_to_load == None:
			filenames = glob('%s/*.freq.bz2' % WC_DIR)
		else:
			filenames = []
			for m in models_to_load:
				if not self.models.has_key(m):
					m = m.replace('/', '_')
				filenames.append( '%s/%s.freq.bz2' % (WC_DIR, m) )

		self.models = {}
		for filename in filenames:
			nm = NounModel()
			fin = BZ2File(filename, 'r')
			nm.load_from_string(fin.readline())
			self.models[nm.noun] = nm
			fin.close()
		print 'Done loading models.'

	def _init_models_with_key(self, key):
		if not self.models.has_key(key):
			self.models[key] = NounModel(key)

	def _lower_lemma_verb(self, verb):
		verb = verb.lower()
		mverb = wn.morphy(verb, pos=nlwn.VERB)
		if mverb:
			return mverb
		return verb

	def add_subject_head_of_verb(self, head_noun, verb, credit):
		verb = self._lower_lemma_verb(verb)
		self._init_models_with_key(head_noun)
		self.models[head_noun].add_subject_head_of_verb(verb, credit)

		if not self.verb_subject_contexts_freqs.has_key(verb):
			self.verb_subject_contexts_freqs[verb] = 0
		self.verb_subject_contexts_freqs[verb] += credit

	def add_object1_head_of_verb(self, head_noun, verb, credit):
		verb = self._lower_lemma_verb(verb)
		self._init_models_with_key(head_noun)
		self.models[head_noun].add_object1_head_of_verb(verb, credit)

		if not self.verb_object1_contexts_freqs.has_key(verb):
			self.verb_object1_contexts_freqs[verb] = 0
		self.verb_object1_contexts_freqs[verb] += credit

	def add_object2_head_of_verb(self, head_noun, verb, credit):
		verb = self._lower_lemma_verb(verb)
		self._init_models_with_key(head_noun)
		self.models[head_noun].add_object2_head_of_verb(verb, credit)

		if not self.verb_object2_contexts_freqs.has_key(verb):
			self.verb_object2_contexts_freqs[verb] = 0
		self.verb_object2_contexts_freqs[verb] += credit

	def add_modifier_of_head(self, head_noun, modifier, credit):
		self._init_models_with_key(head_noun)
		self.models[head_noun].add_modifier_of_head(modifier, credit)

	def add_modified_head(self, modifier, head_noun, credit):
		self._init_models_with_key(modifier)
		self.models[modifier].add_modified_head(head_noun, credit)



	def add_prep_head_noun_of_pp_pair(self, head_noun, prep, head_noun_of_pp, credit):
		self._init_models_with_key(head_noun)
		self.models[head_noun].add_prep_head_noun_of_pp_pair(prep, head_noun_of_pp, credit)

		if not self.prep_head_noun_of_pp_pairs_contexts_freqs.has_key( (prep, head_noun_of_pp) ):
			self.prep_head_noun_of_pp_pairs_contexts_freqs[ (prep, head_noun_of_pp) ] = 0
		self.prep_head_noun_of_pp_pairs_contexts_freqs[ (prep, head_noun_of_pp) ] += credit


	def add_prep_head_noun_pair(self, head_noun_of_pp, prep, head_noun, credit):
		self._init_models_with_key(head_noun_of_pp)
		self.models[head_noun_of_pp].add_prep_head_noun_pair(prep, head_noun, credit)

		if not self.prep_head_noun_pairs_contexts_freqs.has_key( (prep, head_noun) ):
			self.prep_head_noun_pairs_contexts_freqs[ (prep, head_noun) ] = 0
		self.prep_head_noun_pairs_contexts_freqs[ (prep, head_noun) ] += credit

	def add_prep_verb_pair(self, stem, prep, verb, credit):
		verb = self._lower_lemma_verb(verb)
		self._init_models_with_key(stem)
		self.models[stem].add_prep_verb_pair(prep, verb, credit)

		if not self.prep_verb_pairs_contexts_freqs.has_key( (prep, verb) ):
			self.prep_verb_pairs_contexts_freqs[ (prep, verb) ] = 0
		self.prep_verb_pairs_contexts_freqs[ (prep, verb) ] += credit

	def add_cohead(self, head, cohead, credit):
		self._init_models_with_key(head)
		self.models[head].add_cohead(cohead, credit)

	def update_coheads(self, heads, modifiers):
		for head1_num, head1 in enumerate(heads):
			for head2_num, head2 in enumerate(heads):
				if head1_num != head2_num:
					self.add_cohead(head1, head2, 1.0)

		for head1_num, head1 in enumerate(heads):
			for mpos, mstem in modifiers:
				if mpos == 'N':
					self.add_cohead(mstem, head1, 1.0)

	def update_word_count(self, parsetree):

		self.word_count += len(tr.leaves_string(parsetree).split())
		heads = []
		modifiers = []

		word_mod_pairs = modifiers_and_head_nouns(parsetree)
		for word, mod in word_mod_pairs:

			word_list = [word]
			for m in mod:
				word_list.append(m)

			for w_num, w in enumerate(word_list):
				if w[-1].pos in ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VB', 'RP']:
					continue

				pos, word_stems = pos_list_to_stems(w)
				if pos == 'N' and len(word_stems) > 0:
					credit = 1.0 / len(word_stems)
					for stem in word_stems:
						if not self.noun_freqs.has_key(stem):
							self.noun_freqs[stem] = 0
						self.noun_freqs[stem] += credit
						self.all_noun_context_count += credit
						if w_num == 0:
							if not self.head_noun_context_freqs.has_key(stem):
								self.head_noun_context_freqs[stem] = 0
							self.head_noun_context_freqs[stem] += credit
							heads.append(stem)



			for m in mod:
				mpos, mod_stems = pos_list_to_stems(m)
				if len(mod_stems) == 0:
					mpos = m[0].node
					mod_stems = [m[0][0].lower()]

				wpos, word_stems = pos_list_to_stems(word)
				if len(word_stems) > 0:
					credit = 1.0 / (len(word_stems) * len(mod_stems))
					for wstem in word_stems:
						for mstem in mod_stems:
							modifier = (mpos, mstem)
							if mpos not in ['N', 'ADJ']: continue
							modifiers.append(modifier)
							self.add_modifier_of_head(wstem, modifier, credit)

							if mpos == 'N':
								self.add_modified_head(mstem, wstem, credit)

								if not self.modified_head_contexts_freqs.has_key(wstem):
									self.modified_head_contexts_freqs[wstem] = 0
								self.modified_head_contexts_freqs[wstem] += credit

							if not self.modifier_contexts_freqs.has_key(modifier):
								self.modifier_contexts_freqs[modifier] = 0
							self.modifier_contexts_freqs[modifier] += credit

		self.update_coheads(heads, modifiers)

	def update_models_with_np_scope(self, scope):
		np_head_list = scope['NOUN-PHRASE'][0]['HEAD_LIST']
		if len(np_head_list) == 0:
			return

		for pp in scope['PREP-PHRASES']:
			prep = tr.prep_of_pp(pp['TREE'])
			if not prep: continue
			prep = tr.leaves_string(prep).lower()

			for pp_head in pp['HEAD_LIST']:
				credit = 1.0 / len(pp_head['STEMS'])

				for pp_head_stem in pp_head['STEMS']:
					for np_head in np_head_list:
						credit *= 1.0 / len(np_head['STEMS'])
						for np_head_stem in np_head['STEMS']:
							self.add_prep_head_noun_pair(pp_head_stem, prep, np_head_stem, credit)
							self.add_prep_head_noun_of_pp_pair(np_head_stem, prep, pp_head_stem, credit)


	def update_models_with_scope(self, scope):
		if scope['SCOPE-TYPE'] != 'VERB':
			self.update_models_with_np_scope(scope)
			return

		if scope.has_key('PARTICLE'):
			verb = '%s %s' % (tr.leaves_string(scope['VERB'][0]), tr.leaves_string(scope['PARTICLE'][0]))
			mverb = wn.morphy(verb, pos=nlwn.VERB)
			if wn.morphy(verb, pos=nlwn.VERB):
				verb = mverb
			else:
				verb = tr.leaves_string(scope['VERB'][0])
		else:
			verb = tr.leaves_string(scope['VERB'][0])

		if not nlwn.synsets(verb,'v'): return

		for pp_num, pp in enumerate(scope['PREP-PHRASES']):
			for head in pp['HEAD_LIST']:
				credit = 1.0 / len(head['STEMS']) 
				prep = tr.prep_of_pp(pp['TREE'])
				if not prep: continue
				prep = tr.leaves_string(prep).lower()
				
				for stem in head['STEMS']:
					self.add_prep_verb_pair(stem, prep, verb, credit)


		if tr.constituent_has_head_stems(scope, 'SUBJECT'):
			for head in scope['SUBJECT'][0]['HEAD_LIST']:
				credit = 1.0 / len(head['STEMS'])
				for stem in head['STEMS']:
					if scope['VOICE'] != 'PASSIVE':
						self.add_subject_head_of_verb(stem, verb, credit) 
					else:
						self.add_object1_head_of_verb(stem, verb, credit) 

		if scope['VOICE'] != 'PASSIVE' and tr.constituent_has_head_stems(scope, 'OBJECTS', 0):
			for head in scope['OBJECTS'][0]['HEAD_LIST']:
				credit = 1.0 / len(head['STEMS'])
				for stem in head['STEMS']:
					self.add_object1_head_of_verb(stem, verb, credit) 

		if tr.constituent_has_head_stems(scope, 'OBJECTS', 1):
			for head in scope['OBJECTS'][1]['HEAD_LIST']:
				credit = 1.0 / len(head['STEMS'])
				for stem in head['STEMS']:
					self.add_object2_head_of_verb(stem, verb, credit) 


def save_synset_scores(wc_dir):
	models = NounModels()
	models.load_models(wc_dir)
	models.calculate_and_save_synset_scores(wc_dir)

def save_pmi(wc_dir, pmi_dir):
	models = NounModels()
	models.load_models(wc_dir)
	print 'Writing PMI models to directory %s' % pmi_dir
	models.calculate_and_save_pmi(pmi_dir)
	print 'Done writing PMI models.'


def feature_dict_to_model(word_model, feature_dict):
	for k, v in feature_dict.items():
		feature_num, feature_name = k
		feature_type_name = feature_num_to_names[feature_num]
		word_model.__dict__[feature_type_name][feature_name] = v


cached_models = {}
def disambiguate_noun_model(word_model, nounmodels_dir, synset_dir, csim):
	global cached_models

	word_model_feature_dict = word_model.feature_dict()
	
	synset_scores = {}

	word_model_pmi_dict = csim.get_pmi_feature_dict(word_model.noun, word_model_feature_dict)

	general_word_model = PMINounModel()
	general_word_model.noun = word_model.noun
	feature_dict_to_model(general_word_model, word_model_pmi_dict)
	gpmi.generalize_pmi_nounmodel(general_word_model)
	word_model = general_word_model
	word_model_feature_dict = word_model.feature_dict()
	word_model_pmi_dict = word_model_feature_dict

	nsd_debugger.write_word_features(word_model.noun, map_feature_dict_to_names(word_model_pmi_dict))

	synsets = nlwn.synsets(word_model.noun, 'n')
	if len(synsets) == 0: return []
	if len(synsets) == 1: return [(1, synsets[0].name)]

	for synset in synsets:
		synset_scores[synset.name] = 0
		nsd_debugger.start_big_table('Concept vector of %s' % synset.name, colspan=2)

		if cached_models.has_key(synset.name):
			synset_model = cached_models[synset.name]
		else:
			synset_model = PMINounModel(synset.name, synset_dir)
			cached_models[synset.name] = synset_model

		synset_scores[synset.name] += synset_model.cosine_similarity_given_other_feature_dict(word_model_pmi_dict)
		nsd_debugger.finish_big_table()


	sorted_scores = sorted([(synset_scores[synset], synset) for synset in synset_scores], reverse=True, key=operator.itemgetter(0))
	return sorted_scores



def disambiguate_nouns_from_parse(parse, synset_dir, nounmodels_dir, models):
	nsd_debugger.start_new()
	answers = {}
	csim = ContextSimilarity(models)

	sentence, parsetree, scopes = parse


	models.models = {}
	models.update_word_count(parsetree)
	for scope in scopes:
		models.update_models_with_scope(scope)

	for word_model in models.models.values():
		try:
			nsd_debugger.start_big_table('*** %s ***' % word_model.noun, colspan=1, bgcolor='purple', fgcolor='white')
			nsd_debugger.finish_big_table()
			sorted_scores = disambiguate_noun_model(word_model, nounmodels_dir, synset_dir, csim)

			answers[word_model.noun] = sorted_scores[0][1]

		except ZeroDivisionError: print 'zero ic?'
		except IOError: print 'IO Error'
		except KeyError: print '********* key Error', word_model.noun

	nsd_debugger.finish()
	return tuple([(word, answers[word]) for word in answers])

def main():
	if len(sys.argv) == 3 and sys.argv[1] == '-pp':
		pretty_print_nounmodel(sys.argv[2])
		return

	if len(sys.argv) == 3 and sys.argv[1] == '-synsets':
		save_synset_scores(sys.argv[2])
		return

	if len(sys.argv) == 4 and sys.argv[1] == '-savepmi':
		save_pmi(sys.argv[2], sys.argv[3])
		return

	if len(sys.argv) == 4 and sys.argv[1] == '-recountpp':
		recount_pp(sys.argv[2], sys.argv[3])
		return

	if len(sys.argv) == 4 and sys.argv[1] == '-sim':
		make_synset_sim_matrix(sys.argv[2], sys.argv[3])
		return

	if len(sys.argv) == 4 and sys.argv[1] == '-filter':
		filter_contexts(sys.argv[2], sys.argv[3])
		return

	if len(sys.argv) == 4 and sys.argv[1] == '-monorel':
		make_monosemous_relatives(sys.argv[2], sys.argv[3])
		return

	if len(sys.argv) == 5 and sys.argv[1] == '-polyrel':
		make_polysemous_relatives(sys.argv[2], sys.argv[3], sys.argv[4])
		return

	elif len(sys.argv) == 5 and sys.argv[1] == '-pmi':
		stem1 = sys.argv[2]
		stem2 = sys.argv[3]
		print_pmi(stem1, stem2, sys.argv[4])
		return

	elif len(sys.argv) == 5 and sys.argv[1] == '-dis':
		disambiguate_nouns(sys.argv[2], sys.argv[3], sys.argv[4])
		return
	
	elif len(sys.argv) == 4 and sys.argv[1] == '-pr':
		precision_recall(sys.argv[2], int(sys.argv[3]))
		return

	elif len(sys.argv) == 5 and sys.argv[1] == '-merge':
		indir1 = sys.argv[2]
		indir2 = sys.argv[3]
		outdir = sys.argv[4]
		inmodel1 = NounModels()
		inmodel2 = NounModels()
		outmodel = NounModels()

		inmodel1.load_models(indir1)
		inmodel2.load_models(indir2)
		inmodel1.merge(inmodel2)
		inmodel1.save_models_to_files(outdir)
		return

	elif len(sys.argv) == 3 and sys.argv[1] == '-count':
		wc_dir = sys.argv[2]

	else:
		print sys.argv[0], '-pmi <stem1> <stem2> <wc-dir to read>'
		print sys.argv[0], '-pp <wc-file to read>'
		print sys.argv[0], '-count <wc-dir to write>'
		print sys.argv[0], '-merge <in wc-dir 1> <in wc-dir 2> <out wc-dir>'
		print sys.argv[0], '-savepmi <wc-dir to write> <pmi-dir to write>'
		print sys.argv[0], '-monorel <pmi wc-dir to read> <pmi synset dir to write>'
		print sys.argv[0], '-polyrel <pmi wc-dir to read> <pmi synset dir to read> <pmi synset dir to write>'
		print sys.argv[0], '-sim <smpi synsets to read> <matrix_out_filename>'
		print sys.argv[0], '-dis <testing parse dir> <synsets dir> <wc dir>'
		return

	elapsed = 0
	completed_files = 0
	models = NounModels()
	try:
		models.load_models(wc_dir)
	except:
		print 'Failed to load old models'

	parsefiles = glob('../parses/*bz2')
	done_files = []
	total_files = len(parsefiles)
	for filename in parsefiles:
		json_in = BZ2File(filename)

		start = time.time()
		print 'starting %s' % filename
		for line in json_in:
			try:
				parse = savep.parse_json_decode(line)
				sentence, parsetree, scopes = parse

				models.update_word_count(parsetree)
				for scope in scopes:
					models.update_models_with_scope(scope)
			except Exception, e:
				save_traceback(e)

		json_in.close()
		done_files.append(filename)

		print 'finished %s' % filename
		end = time.time()
		elapsed += (end - start)
		completed_files += 1
		print '%d files took %d seconds' % (completed_files, elapsed)
		left = total_files - completed_files
		one_over_the_rate = float(elapsed) /  completed_files
		time_left = left * one_over_the_rate
		print 'TIME LEFT: %d seconds, or %f minutes, or %f hours' % (time_left, time_left / 60.0, time_left / (60.0**2))

		if len(done_files) == 5:
			models.save_models_to_files(wc_dir)
			for s in done_files:
				dest = '%s-DONE' % s
				shutil.move(s, dest)
				print 'moved %s to %s' % (s, dest)

			done_files = []

	if len(done_files) != 0:
		for s in done_files:
			dest = '%s-DONE' % s
			shutil.move(s, dest)
			print 'moved %s to %s' % (s, dest)
		models.save_models_to_files(wc_dir)

if __name__ == "__main__":
	main()
