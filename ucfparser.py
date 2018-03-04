#!/usr/bin/env python
from __future__ import with_statement
import socket
import time
import tempfile
import simplejson
import os
import StringIO
import pdb
import sys
import re
from contextlib import nested
from subprocess import Popen, PIPE
import nltk.tree as nltr
import trees as tr
import heads as hd
import wordnet as wn
import pronouns as pn
import saveparses as savep
from nltk.corpus import wordnet as nlwn
import traceback

honorifics = ['a', 'adj', 'adm', 'adv', 'asst', 'b', 'bart', 'bldg', 'brig', 'bros', 'c', 'capt', 'cmdr', 'col', 'comdr', 'con', 'cpl', 'd', 'dr', 'e', 'ens', 'f', 'g', 'gen', 'gov', 'h', 'hon', 'hosp', 'i', 'insp', 'j', 'k', 'l', 'lt', 'm', 'm', 'mm', 'mr', 'mrs', 'ms', 'maj', 'messrs', 'mlle', 'mme', 'mr', 'mrs', 'ms', 'msgr', 'n', 'o', 'op', 'ord', 'p', 'pfc', 'ph', 'prof', 'pvt', 'q', 'r', 'rep', 'reps', 'res', 'rev', 'rt', 's', 'sen', 'sens', 'sfc', 'sgt', 'sr', 'st', 'supt', 'surg', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'v', 'vs', 'a.m', 'p.m']

honorifics_map = {'asst.': 'assistant', 'bldg.': 'building', 'capt.': 'captain', 'cmdr.': 'commander', \
        'col.': 'colonel', 'comdr.': 'commander', 'dr.': 'doctor', 'gen.' : 'general', 'gov.' : 'governor', \
        'lt.': 'lieutenant', 'mr.': 'mister', 'mrs.' : 'mrs', 'ms.': 'mrs', 'prof.' : 'prof', \
        'a.m.' : 'clock time', 'p.m.' : 'clock time'}

have_regex_str = '( |^)(have|has|had|having)[;., ]'
eat_regex_str = '( |^)(eat|ate|eating|eaten)[;., ]'

programs = {\
        'ne': '/home/michael/workspace/NEpackage1.2/server/NEClassifier-server.pl 3000 3001 3002 %s 0 -w 0',\
        'tokenizer': '/home/michael/workspace/NEpackage1.2/wordsplitter/word-splitter.pl',
        'splitter': '/home/michael/workspace/NEpackage1.2/sentence-boundary/sentence-boundary.pl -d /home/michael/workspace/NEpackage1.2/sentence-boundary/HONORIFICS -i "%s" -o "%s"'}

def remove_heads_from_scope(scope):
    new_scope = {}
    for consti_name in scope:
        if consti_name in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES']:
            new_scope[consti_name] = []
            for consti in scope[consti_name]:
                new_scope[consti_name].append( consti['TREE'] )
        elif not isinstance(consti_name, tuple):
            new_scope[consti_name] = scope[consti_name]
    return new_scope

def remove_heads_from_parse(parse):
    sentence, tree, scopes = parse
    new_scopes = []
    for scope in scopes:
        new_scopes.append(remove_heads_from_scope(scope))
    return [sentence, tree, new_scopes]

def recompute_heads_from_parse(parse):
    #print parse_str_repr(parse)
    parse = remove_heads_from_parse(parse)
    sentence, tree, scopes = parse
    for scope in scopes:
        scope['SCOPE-TYPE'] = 'VERB'
        split_pps(scope)
    scopes = add_np_pp_scopes(tree, scopes)
    scopes = hd.choose_head_nps(tree, scopes, [])
    parse = [sentence, tree, scopes]
    #print '--------------------------------------'
    #print parse_str_repr(parse)
    return parse

def recompute_json_parse(json_parse):
    parse = savep.parse_json_decode(json_parse)
    return savep.parse_json_encode(recompute_heads_from_parse(parse))

class ucfparser:
    """ Python interface class to UCF parser lisp server."""

    def __init__(self, lispserver_sock_name = '/tmp/lispserver.sock'):
        """ lispserver_sock_name: the unix socket name used to communicate with
        the running lisp server. """
        
        # create a probably unique filename for the domain socket.
        # tempfile.mktemp does something similar, but it is deprecated. It is
        # possible either way that this name is not unique by the time we create
        # the socket (even if it is unique right now). If someone has beat us to
        # the filename, bind will simply raise a socket.error.
        self.tempfilename = '%s/ucfparser-py-%s.sock' % (tempfile.gettempdir(), 
                str(time.time()))

        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(self.tempfilename)
        self.socket.connect(lispserver_sock_name)
        
    def parse(self, sentence):
        """Parses sentence string and returns a sentenceparse instance.  Quotes
        " seem to give json errors (a ValueError) and paranthesis () sometimes
        cause the lisp-server parser to break, so filter out sentences with
        these characters. """
        self.socket.sendall(sentence)
        jsondata = simplejson.loads(' '.join(self.socket.recv(8192).split()))
        tree = nltr.bracket_parse(jsondata['parse'])
        scopes = jsondata['scopes']
        scopes = fix_scopes(tree, scopes)
        print tree
        return [sentence, tree, scopes]

    def close(self):
        self.socket.close()
        os.unlink(self.tempfilename)

def parse_get_sentence(parse): return parse[0]
def parse_get_tree(parse): return parse[1]
def parse_get_scopes(parse): return parse[2]

def verb_scopes(scopes):
    for scope in scopes:
        if not scope.has_key('SCOPE-TYPE') or scope['SCOPE-TYPE'] == 'VERB':
            yield scope

def np_scopes(scopes):
    for scope in scopes:
        if scope.has_key('SCOPE-TYPE') and scope['SCOPE-TYPE'] == 'NP':
            yield scope

def fix_scopes(parsetree, scopes):
    # scopes contains indices into parsetree; go through and replace these
    # index entries with Tree instances from parsetree. Note that entries
    # such as VERB, which contain a single integer outside of a list, also
    # become a list with a single subtree, for consistancy.
    subtree_list = [subtree for subtree in parsetree.subtrees()]
    for scope in scopes:
        for infokey in scope:
            if isinstance(scope[infokey], list):
                scope[infokey] = [subtree_list[treeidx]\
                        for treeidx in scope[infokey]\
                        if isinstance(treeidx, int)]
            elif isinstance(scope[infokey], int):
                scope[infokey] = [subtree_list[scope[infokey]]]

    # guarantee some keys exist for easier access
    for scope in scopes:
        if not scope.has_key('SUBJECT'): scope['SUBJECT'] = []
        if not scope.has_key('OBJECTS'): scope['OBJECTS'] = []
        if not scope.has_key('PREP-PHRASES'): scope['PREP-PHRASES'] = []
        if not scope.has_key('VERB'): scope['VERB'] = []
        if not scope.has_key('VOICE'): scope['VOICE'] = ''
        if not scope.has_key('MODIFIERS'): scope['MODIFIERS'] = ''
        if not scope.has_key('CLAUSE-TYPE'): scope['CLAUSE-TYPE'] = ''
        if not scope.has_key('SCOPE-TYPE'): scope['SCOPE-TYPE'] = 'VERB'
        if not scope.has_key('NOUN-PHRASE'): scope['NOUN-PHRASE'] = []


    # make trees immutable so that they are hashable
    for scope in scopes:
        for infokey in scope:
            if infokey in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES', 'VERB', 'MODIFIERS']:
                scope[infokey] = [tree.freeze() for tree in scope[infokey]]

    for scope in scopes:
        split_pps(scope)

    scopes = add_np_pp_scopes(parsetree, scopes)
    return scopes

# a very specific hack to split PP's of the form (PP (PP ..) (PP .. ))
def split_pps(scope):
    pp_replace = {}
    new_pp_trees = []
    for pp_tree in scope['PREP-PHRASES']:
        split_pp = True
        for child in pp_tree:
            if child.node == 'PP':
                new_pp_trees.append(child)
            else:
                split_pp = False
                break
        if split_pp and new_pp_trees:
            pp_replace[pp_tree] = new_pp_trees
    for tree in pp_replace:
        scope['PREP-PHRASES'].remove(tree)
        scope['PREP-PHRASES'].extend(pp_replace[tree])

def candidate_np_pp_pairs(parsetree, scopes):
    exclude_pps = []
    for scope in scopes:
        for pp in scope['PREP-PHRASES']:
            exclude_pps.append(pp)
    
    for subtree in parsetree.subtrees():
        np = None
        attached_pps = []

        for subtree_child in subtree:
            if not isinstance(subtree_child, nltr.Tree): continue

            if subtree_child.node in ["NP", "NP-COORD"]:
                if np and attached_pps:
                    yield (np, attached_pps)
                np = subtree_child
                attached_pps = []

            if subtree_child.node == "PP" and np and subtree_child not in exclude_pps:
                attached_pps.append(subtree_child)

        if np and attached_pps:
            yield (np, attached_pps)


def head_np_consti_of_np_subtree(parsetree):
    np_consti = None
    head_np_tree_list = hd.head_np_of_np(parsetree)
    head_list = []
    for head_np_tree in head_np_tree_list:
        candidate_heads = hd.head_np_candidate_heads(head_np_tree)
        head = hd.choose_head_np_from_candidates(candidate_heads)
        if head:
            head_list.append(head)

    np_consti = hd.make_constituent(parsetree, head_list)
    return np_consti

def add_np_pp_scopes(parsetree, scopes):
    for np_tree, pp_trees in candidate_np_pp_pairs(parsetree, scopes):

        np_consti = head_np_consti_of_np_subtree(np_tree)
        if not np_consti:
            continue

        pp_consti_list = []

        for pp_tree in pp_trees:
            pp_consti = None
            np_tree = hd.np_of_pp(pp_tree)
            if np_tree:
                pp_consti = head_np_consti_of_np_subtree(np_tree)
                if pp_consti:
                    pp_consti['TREE'] = pp_tree
                    pp_consti_list.append(pp_consti)

        if pp_consti_list:
            scope = {'SCOPE-TYPE': 'NP', \
                    'NOUN-PHRASE': [np_consti], \
                    'PREP-PHRASES': pp_consti_list }
            scopes.append(scope)
    return scopes




def xreadlines_parse_filter(fin, parser, verb=None, regex_match_str=None,
        regex_nomatch_str=r'[{}_:;?|&)(><=-]|\[|\]|\"', exclude_phrase=None):

    regex_nomatch = None
    if regex_nomatch_str:
        regex_nomatch = re.compile(regex_nomatch_str)

    if regex_match_str:
        regex_match = re.compile(regex_match_str)

    if exclude_phrase:
        exclude_phrase_regex = re.compile("( |^)%s(;|\.|,| |$)" % exclude_phrase.lower())

    for line in fin.xreadlines():

        if (regex_match_str and not regex_match.search(line)) or \
                (regex_nomatch and regex_nomatch.search(line)):
            continue

        if exclude_phrase and exclude_phrase_regex.search(line.lower()):
            continue

        # some checks to avoid hanging the parser
        if len([c for c in line.split() if c.endswith(",")]) >= 10:
            continue
        if len(line.split()) > 100:
            continue
        if line.find('.....') != -1:
            continue
        if line.find('!!!!!') != -1:
            continue

        # need to tokenize semicolons and colons for the parser, since it doesn't
        # do that itself. It does tokenize other characters itself.
        line = re.sub('(?<=\S)([;:])', r' \1', line) 

        twodots_re = re.compile(r'(\w+) \. (\w+)( |$)(\.|%|\s|$)')
        line = twodots_re.sub(r'\1.\2\4', line)
        percent_re = re.compile(r'([\d]*\.?[\d]+)\s?%')
        line = percent_re.sub(r'\1 percent', line)
        
        line = line.lower()
        for honorific in honorifics:
            line = line.replace(' %s .' % honorific, ' %s. ' % honorific)

        for honorific in honorifics_map:
            line = line.replace(' %s ' % honorific, ' %s ' % honorifics_map[honorific])

        try:
            print line
            parse = parser.parse(line)
            if not verb:
                yield parse
            else:
                for scope in verb_scopes(parse_get_scopes(parse)):
                    if wn.morphy(tr.leaves_string(scope['VERB'][0]).lower(), pos=nlwn.VERB) == verb and \
                            len(scope['SUBJECT']) == 1 and len(scope['OBJECTS']) >= 1:
                        yield parse
                        break
        except ValueError, e:
            sys.stderr.write('ValueError: %s\n' % str(e))
            sys.stderr.write('was parsing: %s' % line)
            traceback.print_exc()

def netag_tokenized(infilename, outfilename):
    fout = open(outfilename, 'w')
    filename = programs['ne'] % infilename
    p = Popen(filename, shell=True, stdout=fout, stderr=PIPE)
    os.waitpid(p.pid, 0)
    fout.close()

def tokenize(infilename, outfilename):
    fout = open(outfilename, 'w')
    p = Popen([programs['tokenizer'], infilename], stdout=fout)
    os.waitpid(p.pid, 0)
    fout.close()

def sentence_splitter(infilename, outfilename):
    filename = programs['splitter'] % (infilename, outfilename)
    p = Popen(filename, shell=True, stderr=PIPE)
    os.waitpid(p.pid, 0)

class ne_goto(Exception): pass
def named_entities(sentences_fname):
    """
    sentences_fname is a file with one sentence on each line. Returns:
    1. a list with keys as parse/sentence numbers, with entries of lists of tuples (ne, ne_type)
    in the order found in that sentence
    2. the filename of the tokenized file generated from the sentences file
    3. the filename of the named entity tagged file generated from the tokenized file
    """

    tokenized_fname = 'tokenized-%s' % sentences_fname
    netagged_fname = 'netagged-%s' % sentences_fname
    tokenize(sentences_fname, tokenized_fname)
    netag_tokenized(tokenized_fname, netagged_fname)

    ne_lists = []

    with nested(open(tokenized_fname, 'r'), open(netagged_fname, 'r')) as (sent_file, netagged_file):

        netagged = netagged_file.read().split()
        netagged_iter = iter(netagged)

        sentences = [line.split() for line in sent_file.read().splitlines()]
        sent_line_num = 0
        sent_tok_idx = 0

        ne_lists = [[] for sent in sentences] 

        try:
            while 1:

                try:
                    # find a named entity in netagged
                    ne_token = netagged_iter.next()
                    if ne_token in ['[PER', '[MISC', '[LOC', '[ORG']:
                        ne = []
                        ne_type = ne_token[1:]
                        ne_token = netagged_iter.next()
                        while ne_token != ']':
                            ne.append(ne_token)
                            ne_token = netagged_iter.next()

                        # we found ne, and netagged_iter is now at the ']'
                        # now find named entity ne in sentences

                        ne_found = []
                        while sent_line_num < len(sentences) and ne != ne_found:

                            sent = sentences[sent_line_num]

                            while sent_tok_idx < len(sent) and ne != ne_found:
                                if sent[sent_tok_idx] == ne[0]:
                                    tmp_sent_tok_idx = sent_tok_idx
                                    ne_idx = 0
                                    while tmp_sent_tok_idx < len(sent) \
                                            and ne_idx < len(ne) \
                                            and sent[tmp_sent_tok_idx] == ne[ne_idx]:
                                        ne_found.append(ne[ne_idx])
                                        ne_idx += 1
                                        tmp_sent_tok_idx += 1

                                    if ne_found == ne:
                                        ne_lists[sent_line_num].append((' '.join(ne), ne_type))
                                        sent_tok_idx += 1
                                        raise ne_goto
                                    else:
                                        ne_found = []

                                sent_tok_idx += 1
                            sent_line_num += 1
                            sent_tok_idx = 0
                except ne_goto:
                    pass

        except StopIteration:
            pass

    return (ne_lists, tokenized_fname, netagged_fname)

def parse_batch(parser, corpus_file, sent_file='sentences.tmp', verb=None, regex=None, batch_size=50, keep_verbs=False, exclude=None, seek=None, filter_chars=True):
    fnum=0

    WITH_NE_SUPPORT = False

    with open(corpus_file, 'r') as fin:

        if seek:
            fin.seek(seek)
            print fin.readline()

        if filter_chars:
            parsegetter = xreadlines_parse_filter(fin, parser, verb, regex, exclude_phrase=exclude)
        else:
            parsegetter = xreadlines_parse_filter(fin, parser, verb, regex, regex_nomatch_str=None, exclude_phrase=exclude)
        eof = False

        while not eof:
            fnum += 1
            parses = []

            if WITH_NE_SUPPORT:
                sent_file_fnum = '%s-%d' % (sent_file, fnum)
                sent_out = open(sent_file_fnum, 'w')

            try:
                for i in xrange(batch_size):
                    parse = parsegetter.next()
                    parses.append(parse)
                    if WITH_NE_SUPPORT:
                        sent_out.write(parse_get_sentence(parse))
            except StopIteration:
                eof = True

            if WITH_NE_SUPPORT:
                sent_out.close()
                ne_lists, tokenized_fname, netagged_fname = named_entities(sent_file_fnum)
                for parse_num, ne_list in enumerate(ne_lists):
                    sentence, tree, scopes = parses[parse_num]
                    scopes = hd.choose_head_nps(tree, scopes, ne_list)
                    parses[parse_num] = [sentence, tree, scopes]

            if not WITH_NE_SUPPORT:
                for parse_num, parse in enumerate(parses):
                    sentence, tree, scopes = parse
                    scopes = hd.choose_head_nps(tree, scopes, [])
                    parses[parse_num] = [sentence, tree, scopes]

            yield filter_parses(parses, verb, keep_verbs)


def filter_parses(parses, verb, keep_verbs):
    # remove bad scopes from the parses.
    filtered_parses = []
    for parse in parses:
        filtered_scopes = []
        sentence, parsetree, scopes = parse
        for scope in np_scopes(scopes):
            filtered_scopes.append(scope)

        for scope in verb_scopes(scopes):
            #if len(scope['SUBJECT']) != 1 or len(scope['OBJECTS']) < 1:
            #    continue

            #if len(scope['SUBJECT']) > 1:
            #    continue

            if not keep_verbs and verb and wn.morphy(tr.leaves_string(scope['VERB'][0]).lower(), pos=nlwn.VERB) != verb:
                continue
            #if scope['CLAUSE-TYPE'] == 'RELATIVE':
            #    continue

            skip = False
            #for infokey in scope:
            #    if isinstance(infokey, tuple) or infokey in ['SUBJECT', 'OBJECTS']:
            #        for constituent in scope[infokey]:
            #            heads = constituent['HEAD_LIST']
            #            if not isinstance(infokey, tuple) and len(heads) == 0:
            #                skip = True
                        #for head in heads:
                        #    if len(head['STEMS']) == 0:
                        #        skip = True
                        #    elif pn.is_pronoun(head['HEAD']) and pn.is_ambiguous_pronoun(head['HEAD']):
                        #        skip = True
            if skip:
                continue

            filtered_scopes.append(scope)

        if len(filtered_scopes) > 0:
            filtered_parse = [sentence, parsetree, filtered_scopes]
            filtered_parses.append(filtered_parse)

    return filtered_parses

def parse_str_repr(parse):
    sentence, parsetree, scopes = parse

    output = StringIO.StringIO()
    output.write('*** SENTENCE: %s\n' % sentence)
    output.write('*** PARSETREE: %s\n' % parsetree)

    for scope in verb_scopes(scopes):

        output.write('\n*** SCOPE OF VERB %s\n' % (scope['VERB'][0].pprint()))

        for infokey in scope:
            if infokey in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES']:
                for constituent in scope[infokey]:
                    tree = constituent['TREE']
                    heads = constituent['HEAD_LIST']
                    output.write('* %s: %s\n' % (infokey, tree.pprint()))
                    if heads:
                        for head in heads:
                            stemstr = ' '.join([stem for stem in head['STEMS']])
                            output.write('|_ HEAD: %s (%s)\n' % (head['HEAD'], stemstr))

            elif isinstance(infokey, tuple):
                print infokey

            elif infokey != 'VERB':
                if isinstance(scope[infokey], list):
                    for tree in scope[infokey]:
                        output.write('* %s: %s\n' % (infokey, tree.pprint()))
                else:
                    output.write('* %s: %s\n' % (infokey, scope[infokey]))

    for scope in np_scopes(scopes):
        output.write('\n*** NP (%s)\n' % (scope['NOUN-PHRASE'][0]['TREE'].pprint()))
        for head in scope['NOUN-PHRASE'][0]['HEAD_LIST']:
            output.write('    HEAD: %s\n' % (head['HEAD']))
        for pp in scope['PREP-PHRASES']:
            output.write('\n^ PP (%s)\n' % (pp['TREE'].pprint()))
            for head in pp['HEAD_LIST']:
                output.write('   HEAD: %s\n' % (head['HEAD']))



    contents = output.getvalue()
    output.close()
    return contents

def get_scopes_from_parses(parses, verb):
    scope_list = []
    for parse in parses:
        sentence, tree, scopes = parse
        for scope in verb_scopes(scopes):
            if wn.morphy(tr.leaves_string(scope['VERB'][0]), pos=nlwn.VERB) == verb:
                scope_list.append(scope)
    return scope_list

