#!/usr/bin/python
from collections import defaultdict

# takes in the discovered concepts file of the form
# ["word1", ["(concept1, x.xx)", "(concept2, x.xx)", ...]]
# ["word2", ["(conceptx, x.xx)", "(concepty, x.xx)", ...]]
# ...
# and output the inverse
# ["concept1", "(word1, x.xx)", "(word2, x.xx)", ...]]
# ...
# 

import simplejson as json
import sys

def create_concept_word_dict(filename):
    concept_word_dict = defaultdict(list)
    concept_word_set = defaultdict(set)

    fin = open(filename)

    while fin:
        word = fin.readline()[6:].strip()
        if len(word) == 0:
            break
        concept_list = json.loads(fin.readline())

        for concept, s in concept_list:
            concept_word_dict[concept].append( (word, s) )

    fin.close()
    return concept_word_dict

def write_concept_word_dict(concept_word_dict, filename):
    fout = open(filename, 'w')
    for concept, word_list in concept_word_dict.items():
        #word_list = [(word, s) for word, s in word_set]
        jsonstr = json.dumps( (concept, word_list) )
        fout.write('%s\n' % (jsonstr,))
    fout.close()

def main():
    readfilename = sys.argv[1]
    writefilename = sys.argv[2]
    d = create_concept_word_dict(readfilename)
    write_concept_word_dict(d, writefilename)
    
if __name__ == "__main__":
    main()

