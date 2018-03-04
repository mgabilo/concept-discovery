#!/usr/bin/python
import ucfparser as up
import saveparses as savep
import websearch as ws
import pdb
import sys

def main():
    if len(sys.argv) != 2:
        print 'arguments: parses-file'
        return


    json_fin_name = sys.argv[1]
    json_in = open(json_fin_name, 'r')
    for line in json_in:
        parse = savep.parse_json_decode(line)
        print up.parse_str_repr(parse)

    json_in.close()

if __name__ == "__main__":
    main()


