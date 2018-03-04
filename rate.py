#!/usr/bin/python

import sys
import view_common_features

filename = 'word-concept-topsim-human-trial'
correct_num = 0
wrong_num = 0

fout = open('rate-out', 'w')

fin = open(filename)
for line in fin:
    redo_count = 0
    read_line = ''

    word, col, concept_str = line.partition(':')
    concept = concept_str.split()[0]
    word = word.strip()

    while read_line.strip() not in ['c', 'w']:
        if redo_count == 0:
            print '\n------------------------------------------------'
            print line
            print 'Answer [c]orrect or [w]rong. Type ? for features.'
            read_line = sys.stdin.readline()
        else:
            print 'Answer [c]orrect or [w]rong. Type ? for features.'
            read_line = sys.stdin.readline()
        redo_count += 1

        if read_line.strip() == '?':
            view_common_features.view_common(word, concept)
    
    if read_line.strip() == 'c':
        correct_num += 1
        fout.write('%s %s c\n' % (word, concept))
    if read_line.strip() == 'w':
        fout.write('%s %s w\n' % (word, concept))
        wrong_num += 1

fout.write('# correct: %d\n' % correct_num)
fout.write('# wrong: %d\n' % wrong_num)
fout.close()

