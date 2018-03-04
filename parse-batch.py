#!/usr/bin/python
import ucfparser as up
import saveparses as savep
import pdb
import sys

def main():
    if len(sys.argv) != 3:
        print 'arguments: corpus-file output-parses-file'
        return


    parser = up.ucfparser()
    fin_name = sys.argv[1]
    json_fin_name = sys.argv[2]
    json_out = open(json_fin_name, 'w')

    for parses in up.parse_batch(parser, fin_name, filter_chars=False):
        for num, parse in enumerate(parses):
            sentence, parsetree, scopes = parse
            try:
                json_repr = savep.parse_json_encode(parse)
                json_out.write(json_repr + '\n')
                json_out.flush()
                parse = savep.parse_json_decode(json_repr)
                sentence, parsetree, scopes = parse
            except UnicodeEncodeError:
                print '*** skipping write parse on unicode error'
            except ValueError:
                print '*** value error?'

            print 'wrote parse %d to %s' % (num, json_fin_name)
            print up.parse_str_repr(parse)
    json_out.close()

if __name__ == "__main__":
    main()


