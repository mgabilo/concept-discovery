#!/usr/bin/python
import ucfparser as up
#import cluster as cl
import saveparses as savep
import websearch as ws
import pdb

def main():
    parser = up.ucfparser()
    fin_name = '/mnt/data/wikipedia/wiki-corpus'
    json_out = open('disambiguated-eat-parses.json', 'w')
    json_out_ambig = open('ambig-parses-3.json', 'w')


    batches = 0

    for parses in up.parse_batch(parser, fin_name, verb='eat', regex=up.eat_regex_str):
        for parse in parses:
            sentence, parsetree, scopes = parse

            parse_has_disambiguated_scopes = False
            for scope in scopes:
                if ws.is_scope_disambiguated(scope):
                    parse_has_disambiguated_scopes = True

            try:
                json_repr = savep.parse_json_encode(parse)
                if parse_has_disambiguated_scopes:
                    json_out.write(json_repr + '\n')
                    json_out.flush()
                else:
                    json_out_ambig.write(json_repr + '\n')
                    json_out_ambig.flush()

            except UnicodeEncodeError:
                pass

    json_out.close()
    json_out_ambig.close()

    

if __name__ == "__main__":
    main()


