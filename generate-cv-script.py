#!/usr/bin/python
from nltk.corpus import wordnet as nlwn
from nltk.corpus.reader.wordnet import information_content
import nltk.corpus.reader.wordnet
import nltk
import sys
import pdb

# sample cv-script entry
#
# ! bond.n.02
# WV monosemouslemma1 monosemouslemma2
# WVI lemma1 lemma2 ; lemma3 lemma4
# P concept.n.01 0.84 ; concept2.n.04 0.95

ic_reader = nltk.corpus.reader.wordnet.WordNetICCorpusReader(nltk.data.find('corpora/wordnet_ic'),'.*\.dat')
ic = ic_reader.ic('ic-bnc-resnik.dat')

HIGH_SIM = 0.7

LOW_SIM =  0.4

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

def hyponyms_by_level(child, level=5):
	levels = []
	levels.append([child])

	for l in xrange(1, level):
		if len(levels) == l:
			levels.append( [] )

		for prev_level_hyp in levels[l-1]:
			prev_hyp = prev_level_hyp.hyponyms() + prev_level_hyp.instance_hyponyms()

			for h in prev_hyp:
				if h.wup_similarity(child) >= HIGH_SIM:
					levels[l].append(h)


		if len(levels[l]) == 0:
			break

	return levels



def hyponyms_by_level_OLD(child, level=5):
	levels = []
	levels.append([child])

	for l in xrange(1, level):
		if len(levels) == l:
			levels.append( [] )

		for prev_level_hyp in levels[l-1]:
			prev_hyp = prev_level_hyp.hyponyms() + prev_level_hyp.instance_hyponyms()

			similar_enough = False
			for h in prev_hyp:
				if h.wup_similarity(child) >= HIGH_SIM:
					similar_enough = True
					break

			if not similar_enough:
				break

			levels[l].extend(prev_hyp)

		if len(levels[l]) == 0:
			break

	return levels



def hyponyms_by_level_FIXED(child, level=5):
	levels = []
	levels.append([child])

	for l in xrange(1, level):
		if len(levels) == l:
			levels.append( [] )

		has_not_too_similar = False

		cand_new_level = []

		for prev_level_hyp in levels[l-1]:
			prev_hyp = prev_level_hyp.hyponyms() + prev_level_hyp.instance_hyponyms()
			cand_new_level.extend(prev_hyp)

			if len(prev_hyp) > 0:

				similar_enough = False
				for h in prev_hyp:
					if h.wup_similarity(child) >= HIGH_SIM:
						similar_enough = True
						break

				if not similar_enough:
					has_not_too_similar = True
					break


		if not has_not_too_similar:
			levels[l].extend(cand_new_level)
		else:
			break

	return levels

def find_intersectable_child_lemmas(base_synset):
	lemma_pairs = []
	child_hyponym_levels = hyponyms_by_level(base_synset)

	for base_lemma in base_synset.lemmas():
		base_lemma_synsets = nlwn.synsets(base_lemma.name(), 'n')
		# ignore monosemous lemmas
		#if len(base_lemma_synsets) <= 1: continue

		noise_synsets = noisy_concepts(base_synset, base_lemma_synsets)

		for child_hyp_level in child_hyponym_levels:
			for child_synset in child_hyp_level:
				for child_lemma in child_synset.lemmas():
					if child_lemma == base_lemma: continue

					lemma_pair_is_noisy = False
					for child_lemma_synset in nlwn.synsets(child_lemma.name(), 'n'):

						if child_synset != child_lemma_synset and synset_is_similar_to_any(child_lemma_synset, noise_synsets):
							lemma_pair_is_noisy = True
							#print 'NOT:', base_synset.name(), base_lemma.name(), child_lemma.name()
					if not lemma_pair_is_noisy:
						lemma_pairs.append( (base_lemma.name(), child_lemma.name()) )


	return set([tuple(sorted(pair)) for pair in lemma_pairs])

def monosemous_lemmas(synset):
	ml = []
	for lemma in synset.lemmas():
		if len(nlwn.synsets(lemma.name(), 'n')) == 1:
			ml.append(lemma.name())
	return ml

def generate_cv(cvscript_filename):
	fout_cv = open(cvscript_filename, 'w')

	# handle entity.n.01 specially since it is the only synset with IC 0; the
	# others that have IC 0 are simply missing/unknown entries. IC 0 entries
	# other than entity.n.01 are handled specially by the algorithm.
	#fout_cv.write('! entity.n.01\nWV entity\nWVI\nP\n\n')

	for synset in nlwn.all_synsets(pos='n'):
		generate_cv_entry(synset, fout_cv)
	fout_cv.close()

def generate_cv_entry(synset, fout):
	#ml = monosemous_lemmas(synset)
	ml = []
	# if synset.name() == 'living_thing.n.01':
	# 	pdb.set_trace()
	intersectable_lemmas = find_intersectable_child_lemmas(synset)

	synset_ic = information_content(synset, ic)

	hypernym_names = []
	for hypernym in synset.hypernyms() + synset.instance_hypernyms():
		hyper_ic = information_content(hypernym, ic)
		if synset_ic == 0 or hyper_ic == 0:
			if hypernym.name != 'entity.n.01':
				shared_ic_ratio = 1.0
			else:
				shared_ic_ratio = 0.0
		else:
			shared_ic_ratio = hyper_ic / synset_ic

		hypernym_names.append((hypernym.name(), shared_ic_ratio))

	fout.write('! %s\n' % synset.name())
	fout.write('WV ')
	for m in ml:
		fout.write('%s ' % m)
	fout.write('\nWVI ')
	for w1, w2 in intersectable_lemmas:
		fout.write('%s %s ; ' % (w1, w2))
	fout.write('\nP ')
	for p, weight in hypernym_names:
		fout.write('%s %f ; ' % (p, weight))
	fout.write('\n\n')

def pairs(elem_list):
	for i in range(len(elem_list)):
		elem1 = elem_list[i]
		for offset, elem2 in enumerate(elem_list[i+1:]):
			j = offset + i + 1
			yield (i, j, elem1, elem2)

def main():
	if len(sys.argv) != 2:
		print 'Syntax: <filename-out>'
		print 'Generates a script to create concept vectors from vector representations of words'
		return


	filename = sys.argv[1]
	generate_cv(filename)

if __name__ == "__main__":
	main()

