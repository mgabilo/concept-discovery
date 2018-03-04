#!/usr/bin/python

import simplejson as json
from nltk.corpus import wordnet as nlwn
import pdb
import nltk.tree as nltr
import ucfparser as up
import heads as hd
import trees as tr

def parses_write(parses, fout):
    for parse in parses:
        fout.write(parse_json_encode(parse) + '\n')

def parse_json_encode(parse):
    sentence, parsetree, scopes = parse
    return json.dumps([sentence, parsetree_json_encode(parsetree), scope_list_json_encode(scopes)])

def parse_json_decode(parse):
    sentence, parsetree, scopes = json.loads(parse)
    return [sentence, parsetree_json_decode(parsetree), scope_list_json_decode(scopes)]

def scope_list_json_encode(scope_list):
    new_scope_list = []
    for scope in scope_list:
        new_scope_list.append(scope_json_encode(scope))
    return new_scope_list

def scope_list_json_decode(scope_list):
    for i in xrange(len(scope_list)):
        scope_list[i] = scope_json_decode(scope_list[i])
    return scope_list

def scope_json_encode(scope):
    new_scope = {}
    for scope_key in scope:
        if isinstance(scope_key, tuple) or scope_key in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES', 'NOUN-PHRASE']:
            new_scope[scope_key_json_encode(scope_key)] = constituent_list_json_encode(scope[scope_key])
        else:
            if isinstance(scope[scope_key], list):
                new_scope[scope_key] = tree_list_json_encode(scope[scope_key])
            else:
                new_scope[scope_key] = scope[scope_key]
    return new_scope

def scope_json_decode(scope):
    new_scope = {}
    for scope_key in scope:
        scope_key_decoded = scope_key_json_decode(scope_key)
        if isinstance(scope_key_decoded, tuple) or scope_key_decoded in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES', 'NOUN-PHRASE']:
            new_scope[scope_key_decoded] = constituent_list_json_decode(scope[scope_key])
        else:
            if isinstance(scope[scope_key], list):
                new_scope[scope_key_decoded] = tree_list_json_decode(scope[scope_key])
            else:
                new_scope[scope_key_decoded] = scope[scope_key]
    return new_scope


def tree_list_json_encode(tree_list):
    new_trees = []
    for tree in tree_list:
        new_trees.append(parsetree_json_encode(tree))
    return new_trees

def tree_list_json_decode(tree_list):
    for i in xrange(len(tree_list)):
        tree_list[i] = parsetree_json_decode(tree_list[i])
    return tree_list

def scope_key_json_encode(scope_key):
    if isinstance(scope_key, tuple):
        consti_type, consti_tree = scope_key
        consti_tree_str = parsetree_json_encode(consti_tree)
        return str((str(consti_type), str(consti_tree_str)))
    return scope_key

def scope_key_json_decode(scope_key):
    if scope_key[0] == '(':
        consti_type, consti_tree_str = [str(s.strip()) for s in scope_key[1:-1].split(",", 1)]
        consti_tree_str = consti_tree_str[1:-1]
        consti_type = consti_type[1:-1]
        consti_tree = parsetree_json_decode(consti_tree_str)
        return (consti_type, consti_tree)
    return scope_key

def constituent_list_json_encode(constituent_list):
    new_const = []
    for constituent in constituent_list:
        new_const.append(constituent_json_encode(constituent))
    return new_const

def constituent_list_json_decode(constituent_list):
    for i in xrange(len(constituent_list)):
        constituent_list[i] = constituent_json_decode(constituent_list[i])
    return constituent_list

def constituent_json_encode(constituent):
    json_tree = parsetree_json_encode(constituent['TREE'])
    json_head_list = head_list_json_encode(constituent['HEAD_LIST'])
    return hd.make_constituent(json_tree, json_head_list)

def constituent_json_decode(constituent):
    constituent['TREE'] = parsetree_json_decode(constituent['TREE'])
    constituent['HEAD_LIST'] = head_list_json_decode(constituent['HEAD_LIST'])
    return constituent

def head_list_json_encode(head_list):
    new_heads = []
    for head in head_list:
        new_heads.append(head_json_encode(head))
    return new_heads

def head_list_json_decode(head_list):
    for i in xrange(len(head_list)):
        head_list[i] = head_json_decode(head_list[i])
    return head_list

def head_json_encode(head):
    wn_synset_str = noun_synset_json_encode(head['SYNSET'])

    new_pos_list = []
    for pos_tree in head['POS_LIST']:
        new_pos_list.append(parsetree_json_encode(pos_tree))
    new_tree = parsetree_json_encode(head['TREE'])
    candidate_head = hd.make_candidate_head(head['HEAD'], new_tree, new_pos_list)

    return hd.make_head(candidate_head, head['STEMS'], wn_synset_str)

def head_json_decode(head):
    head['SYNSET'] = noun_synset_json_decode(head['SYNSET'])
    head['POS_LIST'] = [parsetree_json_decode(pos_tree) for pos_tree in head['POS_LIST']]
    head['TREE'] = parsetree_json_decode(head['TREE'])
    return head


def noun_synset_json_decode(synset_str):
    if synset_str != 'null':
        word, sense = synset_str.split()
        return nlwn.synsets(word)[int(sense)-1]
    return None

def noun_synset_json_encode(synset):
    if synset:
        word, pos, sense = synset.name.rsplit('.',2)
        sense = str('%d' % int(sense))
        return "%s %s" % (word, sense)
    return 'null'

def parsetree_json_decode(parsetree_json):
    if parsetree_json != 'null':
        return nltr.bracket_parse(parsetree_json).freeze()
    return None

def parsetree_json_encode(parsetree):
    if parsetree:
        return parsetree_tostr_rec(parsetree) + ')'
    return 'null'

def parsetree_tostr_rec(parsetree):

    s = '(' + parsetree.node + ' '
    for child in parsetree:
        if isinstance(child, nltr.Tree):
            s += parsetree_tostr_rec(child) + ') '
        else:
            s += child

    return s 

