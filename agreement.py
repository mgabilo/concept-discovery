#!/usr/bin/python

import sys

def main():
	file1 = sys.argv[1]
	file2 = sys.argv[2]

	file1_lines = open(file1).readlines()
	file2_lines = open(file2).readlines()
	assert(len(file1_lines) == len(file2_lines))

	agree_count = 0
	disagree_count = 0

	for line_num, file1_line in enumerate(file1_lines):
		file2_line = file2_lines[line_num]

		file1_mark = file1_line.split()[0]
		file2_mark = file2_line.split()[0]
		assert(file1_mark in ['C', 'W'])
		assert(file2_mark in ['C', 'W'])

		if file1_mark == file2_mark:
			agree_count += 1
		else:
			print 'shrapx:', file1_line
			print 'eno:', file2_line
			print '------'
			disagree_count += 1

	print 'agree count:', agree_count
	print 'disagree count:', disagree_count

	print float(agree_count) / (agree_count + disagree_count)

if __name__ == "__main__":
	main()
