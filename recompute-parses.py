#!/usr/bin/python
import ucfparser as up
import saveparses as savep
import pdb
import sys
from bz2 import BZ2File


def main():
    if len(sys.argv) != 3:
        print 'arguments: input-json-parses-file output-json-parses-file'
        return

    fin_name = sys.argv[1]
    fout_name = sys.argv[2]
    fin = BZ2File(fin_name)
    fout = BZ2File(fout_name, 'w')
    for line in fin:
        fout.write('%s\n' % up.recompute_json_parse(line))
    fout.close()
    fin.close()

if __name__ == "__main__":
    main()

