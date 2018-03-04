#!/usr/bin/python
import pdb
import features
import sys
import os

class CVScriptEntry(object):
	def __init__(self, concept_line, wv_line, wvi_line, p_line):
		self.concept = concept_line[2:]
		self.wv = wv_line[3:].split()
		self.wvi = []
		self.p = []

		for w1_w2 in wvi_line[4:].split(';'):
			w1_w2_tup = w1_w2.split()
			if len(w1_w2_tup) == 2:
				w1, w2 = w1_w2_tup
				self.wvi.append( (w1.strip(), w2.strip()) )

		for parent_weight in p_line[2:].split(';'):
			parent_weight_tup = parent_weight.split()
			if len(parent_weight_tup) == 2:
				parent, weight = parent_weight_tup
				self.p.append( (parent.strip(), float(weight.strip())) )

#def load_cv_script(cv_script_filename='cv-script-nomono'):
def load_cv_script(cv_script_filename):
	cv_script_entries = {}
	fin = open(cv_script_filename)
	while True:
		concept_line = fin.readline().strip()
		if not concept_line:
			break
		wv_line = fin.readline().strip()
		wvi_line = fin.readline().strip()
		p_line = fin.readline().strip()
		fin.readline()
		entry = CVScriptEntry(concept_line, wv_line, wvi_line, p_line)
		cv_script_entries[entry.concept] = entry
		
	return cv_script_entries	

cached_pmi_models = {}
def get_pmi_model_OLD1(wc_dir, noun):
	if cached_pmi_models.has_key(noun):
		return cached_pmi_models[noun]
	else:
		pmi_model = features.PMINounModel()
		filename = pmi_model.lemma_to_filename(noun, wc_dir)
		try:
			os.stat(filename)
		except OSError:
			return None
		pmi_model._load_from_pmi_file(filename)
		cached_pmi_models[noun] = pmi_model
		return pmi_model

def get_pmi_model(wc_dir, noun):
	pmi_model = features.PMINounModel()
	filename = pmi_model.lemma_to_filename(noun, wc_dir)
	try:
		os.stat(filename)
	except OSError:
		return None
	pmi_model._load_from_pmi_file(filename)
	return pmi_model

def generate_cv_entries(cv_entries_dict, input_dir, output_dir):
	nm = features.PMINounModel()
	for concept in cv_entries_dict:
		filename = nm.lemma_to_filename(concept, output_dir)
		concept_exists_already = False

		try:
			os.stat(filename)
			concept_exists_already = True
		except OSError:
			pass

		if not concept_exists_already:
			generate_cv(cv_entries_dict[concept], cv_entries_dict, input_dir, output_dir)

def generate_cv(cv_entry, cv_entries_dict, input_dir, output_dir):
	concept = cv_entry.concept

	# check if it exists on disk already
	pmi_model = features.PMINounModel()
	filename = pmi_model.lemma_to_filename(concept, output_dir)
	try:
		os.stat(filename)
		pmi_model._load_from_pmi_file(filename)
		return pmi_model
	except OSError:
		pass

	# since it doesn't exist on disk, compute it recursively
	wv_models = []

	for word in cv_entry.wv:
		word_pmi_model = get_pmi_model(input_dir, word)
		if word_pmi_model:
			wv_models.append(word_pmi_model)

	for word1, word2 in cv_entry.wvi:
		word1_pmi_model = get_pmi_model(input_dir, word1)
		word2_pmi_model = get_pmi_model(input_dir, word2)
		if word1_pmi_model and word2_pmi_model:
			wv_models.append(word1_pmi_model.intersection(word2_pmi_model))

	#for parent, weight in cv_entry.p:
	#	 parent_cv_entry = cv_entries_dict[parent]
	#	 parent_pmi_model = generate_cv(parent_cv_entry, cv_entries_dict, input_dir, output_dir)
	#	 parent_pmi_model.scale(weight)
	#	 wv_models.append(parent_pmi_model)

	concept_pmi_model = features.PMINounModel()
	concept_pmi_model.noun = concept
	for model in wv_models:
		concept_pmi_model.union_max(model)

	concept_pmi_model.save_to_file(output_dir)
	return concept_pmi_model

# uses cv of the parents
def generate_cv_old_recursive(cv_entry, cv_entries_dict, input_dir, output_dir):
	concept = cv_entry.concept

	# check if it exists on disk already
	pmi_model = features.PMINounModel()
	filename = pmi_model.lemma_to_filename(concept, output_dir)
	try:
		os.stat(filename)
		pmi_model._load_from_pmi_file(filename)
		return pmi_model
	except OSError:
		pass

	# since it doesn't exist on disk, compute it recursively
	wv_models = []

	for word in cv_entry.wv:
		word_pmi_model = get_pmi_model(input_dir, word)
		if word_pmi_model:
			wv_models.append(word_pmi_model)

	for word1, word2 in cv_entry.wvi:
		word1_pmi_model = get_pmi_model(input_dir, word1)
		word2_pmi_model = get_pmi_model(input_dir, word2)
		if word1_pmi_model and word2_pmi_model:
			wv_models.append(word1_pmi_model.intersection(word2_pmi_model))

	for parent, weight in cv_entry.p:
		parent_cv_entry = cv_entries_dict[parent]
		parent_pmi_model = generate_cv(parent_cv_entry, cv_entries_dict, input_dir, output_dir)
		parent_pmi_model.scale(weight)
		wv_models.append(parent_pmi_model)

	concept_pmi_model = features.PMINounModel()
	concept_pmi_model.noun = concept
	for model in wv_models:
		concept_pmi_model.union_max(model)

	concept_pmi_model.save_to_file(output_dir)
	return concept_pmi_model

def main():
	if len(sys.argv) != 4:
		print 'Arguments: <cv-script> <input pmi dir> <output pmi dir>'
		return

	cv_script = sys.argv[1]
	input_dir = sys.argv[2]
	output_dir = sys.argv[3]
	cv_script_entries = load_cv_script(cv_script)
	generate_cv_entries(cv_script_entries, input_dir, output_dir)

if __name__ == "__main__":
	main()
