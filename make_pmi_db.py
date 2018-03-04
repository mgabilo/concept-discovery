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

def make_db(source_dir, c, feature_map_toidx, concept_map_toidx):
	for n, pmi_filename in enumerate(glob.glob('%s/*.pmi.bz2' % source_dir)):
		print n
		pmi_model = features.PMINounModel()
		pmi_model._load_from_pmi_file(pmi_filename)
		concept = pmi_model.noun
		for section in pmi_model.sections:
			for feature in pmi_model.__dict__[section]:
				pmi = pmi_model.__dict__[section][feature]
				feature_str = featuremap.feature_to_str(feature)
				feature_idx = feature_map_toidx[(section, feature_str)]
				concept_idx = concept_map_toidx[concept]

				c.execute('insert into concepts values (?, ?, ?)', (concept_idx, feature_idx, pmi))



def load_words_mem_db(filename, word_map_toidx, word_map_fromidx=None):
	conn_disk = sqlite3.connect(filename)
	c_disk = conn_disk.cursor()
	conn_mem = sqlite3.connect(':memory:')
	conn_mem.execute('''create table words
	(word INTEGER, feature INTEGER, pmi double)''')
	conn_mem.execute('''create index idx_word on words (word)''')
	conn_mem.execute('''create index idx_feature on words (feature)''')
	c_mem = conn_mem.cursor()

	num_loaded = 0
	words_loaded_list = []

	for word in word_map_toidx.values():

		c_disk.execute('select feature, pmi from words where word = ?', (word,))
		n = 0
		num_features = 0
		rows = []
		for feature, pmi in c_disk:
			rows.append( (word, feature, pmi) )
			num_features += 1
			if pmi > 7.0:
				n += 1
		if n > 30:
			words_loaded_list.append((word,n))
			num_loaded += 1
			for word, feature, pmi in rows:
				c_mem.execute('insert into words values (?, ?, ?)', (word, feature, pmi))
	c_mem.close()
	conn_mem.commit()

	print 'loaded # words:', num_loaded
	fout = open('words-loaded-list.txt', 'w')
	words_loaded_list.sort(key=operator.itemgetter(1), reverse=True)
	for word,n in words_loaded_list:
		fout.write('%s %d\n' % (word_map_fromidx[word], n))
	fout.close()

	return conn_mem

def load_concepts_mem_db(filename, concept_map_toidx):
	conn_disk = sqlite3.connect(filename)
	c_disk = conn_disk.cursor()
	conn_mem = sqlite3.connect(':memory:')
	conn_mem.execute('''create table concepts
	(concept INTEGER, feature INTEGER, pmi double)''')
	conn_mem.execute('''create index idx_concept on concepts (concept)''')
	conn_mem.execute('''create index idx_feature on concepts (feature)''')
	c_mem = conn_mem.cursor()

	num_loaded = 0

	for synset in nlwn.all_synsets(pos='n'):

		concept = concept_map_toidx[synset.name()]
		c_disk.execute('select feature, pmi from concepts where concept = ?', (concept,))
		n = 0
		rows = []
		for feature, pmi in c_disk:
			rows.append( (concept, feature, pmi) )
			if pmi > 5.0:
				n += 1
		if n > 15:
			num_loaded += 1
			for concept, feature, pmi in rows:
				c_mem.execute('insert into concepts values (?, ?, ?)', (concept, feature, pmi))

		#if num_loaded == 50:
		#	 break

	c_mem.close()
	conn_mem.commit()
	print 'loaded # concepts:', num_loaded
	return conn_mem
	

def main():
	if len(sys.argv) != 3:
		print 'Arguments: <source pmi dir> <dbfile>'
		return

	source_pmi_dir = sys.argv[1]
	dbfile = sys.argv[2]
	conn = sqlite3.connect(dbfile)
	c = conn.cursor()

	c.execute('''create table concepts
	(concept INTEGER, feature INTEGER, pmi double)''')
	c.execute('''create index idx_concept on concepts (concept)''')
	c.execute('''create index idx_feature on concepts (feature)''')
	feature_map_toidx, feature_map_fromidx = featuremap.load_feature_map()
	concept_map_toidx, concept_map_fromidx = featuremap.load_concept_map()

	make_db(source_pmi_dir, c, feature_map_toidx, concept_map_toidx)

	conn.commit()
	c.close()

if __name__ == "__main__":
	main()

