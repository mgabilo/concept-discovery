#!/usr/bin/env python

import pdb
import trees as tr
import wordnet as wn
import pronouns as pn
import names as na
import ucfparser as up
from nltk.corpus import wordnet as nlwn
import nltk.tree as nltr

ARTICLES = ['a', 'an', 'the']

def make_candidate_head(candidate, tree, pos_list):
    return {'HEAD': candidate, \
            'TREE': tree, \
            'POS_LIST': pos_list}


def make_head(candidate_head, stems, synset):
    return {'HEAD': candidate_head['HEAD'],\
            'STEMS': stems,\
            'SYNSET': synset,\
            'TREE': candidate_head['TREE'],\
            'POS_LIST': candidate_head['POS_LIST']}

def make_constituent(tree, head_list):
    return {'TREE': tree, \
            'HEAD_LIST': head_list}

def freeze_head(head):
    new_head = {}
    new_head['HEAD'] = head['HEAD']
    new_head['STEMS'] = tuple(head['STEMS'])
    new_head['SYNSET'] = head['SYNSET']
    new_head['TREE'] = head['TREE']
    new_head['POS_LIST'] = tuple(head['POS_LIST'])
    return new_head

def split_np_coord(np_coord):
    np_list = []
    for subtree in np_coord:
        if subtree.node == "NP":
            np_list.append(subtree)

    if np_list:
        return np_list

    last_nn = None
    for subtree in np_coord:
        if subtree.node in ['NN', 'NNS']:
            last_nn = subtree
        if subtree.node in ['CC', 'COMMA'] and last_nn:
            np_list.append(last_nn)
            last_nn = None
    if last_nn:
        np_list.append(last_nn)

    return np_list

def head_np_of_np_rule_container(np_subtree):
    """ INPUT:  np_subtree is an NP subtree
        OUTPUT: Head NP subtree or None """

    if np_subtree.node in ["NP", "NP-REL"] \
            and tr.tree_startswith_labels(np_subtree, [['NP'], ['PP']]) \
            and tr.tree_startswith_labels(np_subtree[1], [['IN']]):

        if tr.leaves_string(np_subtree[1][0]).lower() == "of":
            for candidate in head_np_candidate_heads(np_subtree[0]):
                head = candidate['HEAD']
                tree = candidate['TREE']
                stemmed = wn.morphy(head, pos=nlwn.NOUN)
                if stemmed:
                    for container in wn.containers:
                        if stemmed and len(nlwn.synsets(stemmed, pos=nlwn.NOUN)) > 0 and wn.synset_subsumes_word(container, stemmed):
                            return tr.find_first_constituent(np_subtree[1], ['NP', 'NP-COORD'])

        if tr.leaves_string(np_subtree[1][0]).lower() in ["of", "than"]:
            for candidate in head_np_candidate_heads(np_subtree[0]):
                head = candidate['HEAD']
                tree = candidate['TREE']
                stemmed = wn.morphy(head, pos=nlwn.ADJ)
                if stemmed:
                    for synset in nlwn.synsets(stemmed, pos=nlwn.ADJ):
                        if synset.definition.find('quantifier') != -1:
                            return tr.find_first_constituent(np_subtree[1], ['NP', 'NP-COORD'])

def head_np_of_np(np_subtree):
    """ INPUT:  np_subtree is a tree that contains an NP
        OUTPUT: Head NP subtree list """

    if np_subtree.node in ["NN", "NNS", "NNP", "NNPS"]:
        return [np_subtree]

    excluded_subtrees = []

    if np_subtree.node in ["NP", "NP-REL", "NP-COORD"]:
        for subtree in np_subtree.subtrees():

            if subtree.node == "NP-COORD":
                split_np_list = split_np_coord(subtree)
                head_np_list = []
                for split_np in split_np_list:
                    head_np = head_np_of_np(split_np)
                    if head_np:
                        head_np_list.extend(head_np)
                if head_np_list:
                    return head_np_list


            # Exclude certain parts of the tree from consideration
            if subtree.node in ['ADJP', 'VP', 'PP', 'S', 'SBAR']:
                excluded_subtrees.append(subtree)

            skip = False
            for excluded_tree in excluded_subtrees:
                if tr.has_subtree(excluded_tree, subtree):
                    skip = True
            if skip:
                continue

            sub_np = head_np_of_np_rule_container(subtree)
            if sub_np:
                return head_np_of_np(sub_np)


            if subtree.node in ["NP", "NP-COORD"]:

                tree_ends_with_noun = tr.tree_endswith_wordnet_noun(subtree)
                tree_ends_with_pronoun = pn.is_pronoun(tr.leaves_string(subtree[-1]))
                tree_ends_with_nnp = tr.tree_endswith_labels_deep(subtree, [ ['NNP', 'NNPS' ] ])
                
                # we don't trust the tagger too much, so don't put JJ or JJS here
                tree_ends_with_ok_tag = not tr.tree_endswith_labels_deep(subtree, [ ['DT', 'CD' ] ])

                tree_ends_ok = (tree_ends_with_noun or tree_ends_with_pronoun or tree_ends_with_nnp) and tree_ends_with_ok_tag

                # since we got this far, we normally don't want PP's or ADJP in
                # the head NP. The exception is there is a candidate head in
                # subtree that is in wordnet, and the PP that follows does not
                # itself have PP's or ADJP. 
                # * This is only for detecting and allowing wordnet nouns that
                # themselves have a preposition, such as burden_of_proof.
                tree_has_bad_constituents = tr.find_first_constituent(subtree, ['PP', 'ADJP'])
                first_pp = tr.find_first_constituent(subtree, ['PP'])
                wn_noun = wn.has_wordnet_noun([candidate['HEAD'] for candidate in head_np_candidate_heads(subtree)]) 
                if wn_noun and first_pp and wn_noun.endswith(tr.leaves_string(first_pp)):
                    first_np_of_pp = np_of_pp(first_pp)
                    if first_np_of_pp and not tr.find_first_constituent(first_np_of_pp, ['PP', 'ADJP']):
                        tree_has_bad_constituents = False

                if tree_ends_ok and not tree_has_bad_constituents:
                    return [subtree]
    return []

def np_of_pp(pp_subtree):
    """ INPUT:  PP subtree
        OUTPUT: First NP of the PP """
    if pp_subtree.node == "PP":
        for child in pp_subtree:
            if child.node in ["NP", "NP-REL", "NP-COORD"]:
                return child

def head_np_of_pp(pp):
    """ INPUT:  PP subtree
        OUTPUT: Head NP of the first NP of PP """
    np = np_of_pp(pp)
    if np:
        return head_np_of_np(np)


def head_np_candidate_heads(np_subtree):
    """ INPUT: np_subtree is head (simple) NP tree
     OUTPUT: a list of candidate head dicts """

    candidates = []
    words_pos = tr.tree_pos(np_subtree)
    subtree_list = [subtree for subtree in np_subtree.subtrees()]

    for i in xrange(len(words_pos)):
        head_pos_list = words_pos[i:len(words_pos)]
        candidate_head = tr.concat_tree_list_to_string(head_pos_list)

        head_added = False
        for tree in subtree_list:
            if tr.leaves_string(tree) == candidate_head:
                candidates.append(make_candidate_head(candidate_head, tree, head_pos_list))
                head_added = True
                break
        if not head_added:
            candidates.append(make_candidate_head(candidate_head, None, head_pos_list))

    return candidates



def get_candidate_head_nps(parse_subtree):
    """ INPUT: parse_subtree corresponds to one of the subtrees of the OBJECTS,
               SUBJECT or PREP-PHRASES trees from the scopes. None-simple tree.
    OUTPUT: List of (lists of head NP candidate heads) for each head. """

    head_np_list = head_np_of_np(parse_subtree)
    if not head_np_list:
        head_np_list = head_np_of_pp(parse_subtree)

    if head_np_list:
        candidate_heads_list = []
        for head_np in head_np_list:
            candidate_heads_list.append(head_np_candidate_heads(head_np))
        return candidate_heads_list

    return []


def choose_head_np_wordnet(candidate_heads):
    """ INPUT: list of candidate head dicts for a single head
     OUTPUT:   a head dict or None
     checks if there is candidate head that is in wordnet """

    for candidate in candidate_heads:
        stems = wn.morphy2(candidate['HEAD'].lower())
        if len(stems) > 0:
            wn_synset = None
            if len(stems) == 1 and len(nlwn.synsets(stems[0], pos=nlwn.NOUN)) == 1:
                wn_synset = nlwn.synsets(stems[0], pos=nlwn.NOUN)[0]
            return make_head(candidate, stems, wn_synset)

def choose_head_np_ne(candidate_heads, ne_string, ne_type):
    """ INPUT: list of candidate head dicts, named entity string possibly in the
     head, and its type
     OUTPUT: a head dict
     checks if there is a candidate head that matches ne_string """

    for candidate in candidate_heads:
        if ne_string == candidate['HEAD'] and ne_type in ["PER", "LOC", "ORG"]:
            stemmed1 = wn.map_ne_to_stem[ne_type]
            wn_synset = wn.map_ne_to_synset[ne_type]
            return make_head(candidate, [stemmed1], wn_synset)

def choose_head_np_names(candidate_heads):
    for candidate in candidate_heads:
        if na.is_name(candidate['HEAD']):
            return make_head(candidate, [candidate['HEAD'].lower()], wn.person_synset)


def choose_head_np_pronouns(candidate_heads):
    """ INPUT: list of candidate head dicts
     OUTPUT: a head dict
     checks if there is a candidate head with a pronoun to synset mapping """

    for candidate in candidate_heads:
        wn_synset = pn.pronoun_mapper.map(candidate['HEAD'])
        if wn_synset:
            stemmed1 = pn.pronoun_mapper.map_to_stem(candidate['HEAD'])
            return make_head(candidate, [stemmed1], wn_synset)

        if pn.is_ambiguous_pronoun(candidate['HEAD']):
            return make_head(candidate, [candidate['HEAD'].lower()], None)




def choose_head_np_from_candidates(candidate_heads, ne_list=None):
    """ INPUT: list of candidate head dicts corresponding to a single head
     OUTPUT: a head dict """

    # check if pronoun
    heads = choose_head_np_pronouns(candidate_heads)
    if heads:
        return heads

    # weird logic between choosing from ne_list or wordnet synset
    heads_ne = None
    if ne_list:
        for ne_string, ne_type in ne_list:
            if ne_type == "PER":
                heads = choose_head_np_ne(candidate_heads, ne_string, ne_type)
                if heads:
                    heads_ne = heads

    heads_wordnet = choose_head_np_wordnet(candidate_heads)
    if heads_wordnet and not heads_ne:
        return heads_wordnet 
    if not heads_wordnet and heads_ne:
        return heads_ne
    if heads_wordnet and heads_ne:
        if wn.stems_have_instance(heads_wordnet['STEMS']):
            heads_wordnet['STEMS'].extend( heads_ne['STEMS'] )
            return heads_wordnet

    if ne_list:
        for ne_string, ne_type in ne_list:
            heads = choose_head_np_ne(candidate_heads, ne_string, ne_type)
            if heads:
                return heads

    # since it's not in WN, check if it's in the names list
    heads = choose_head_np_names(candidate_heads)
    if heads:
        return heads

    return []

def find_constituent_by_tree(constituent_list, tree):
    for constituent in constituent_list:
        if constituent['TREE'] == tree:
            return constituent

def choose_head_nps(parsetree, scopes, ne_list):
    """ augments up.verb_scopes(scopes) to include head structure,
        returns augmented scopes; prior scope entries assumed to be lists of trees """

    # copy the scopes into augmented_scopes
    augmented_scopes = []
    for scope in scopes:
        augmented_scopes.append( {} )
        for consti_name in scope:
            augmented_scopes[-1][consti_name] = scope[consti_name]

    for scope_num, scope in enumerate(up.verb_scopes(scopes)):
        for consti_name in scope:
            if consti_name in ['SUBJECT', 'OBJECTS', 'PREP-PHRASES']:
                augmented_scope_of_consti = []
                for tree in scope[consti_name]:
                    candidate_heads_lists = get_candidate_head_nps(tree)
                    heads = []
                    for candidate_heads in candidate_heads_lists:
                        head = choose_head_np_from_candidates(candidate_heads, ne_list)
                        if head:
                            heads.append(head)
                    augmented_scope_of_consti.append(make_constituent(tree, heads))
                augmented_scopes[scope_num][consti_name] = augmented_scope_of_consti
    return augmented_scopes

