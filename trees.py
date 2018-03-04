#!/usr/bin/env python
#
# trees.py is a set collection of functions that work on parse trees; that is,
# nltk.tree.Tree instances
#
import heads as hd
import wordnet as wn
import nltk.tree as nltr
import pdb
from nltk.corpus import wordnet as nlwn

def leaves_string(subtree):
    return ' '.join(subtree.leaves())

def tree_pos(tree):
    pos_list = []
    for subtree in tree.subtrees():
        if not isinstance(subtree[0], nltr.Tree):
            pos_list.append(subtree)
    return pos_list

# return a list of NP or NN/NNS/NNP/NNPS subtrees such that there are no
# embedded NP subtrees in any tree
def smallest_np_subtrees(subtree):
    trees = []
    for tree in subtree.subtrees():

        if tree.node in ['NP', 'NP-COORD']:
            can_add = True
            for t in tree[0:]:
                if find_first_constituent(t, 'NP'):
                    can_add = False
            if can_add:
                trees.append(tree)

    return trees

def modifiers_of_head_noun(scope, constituent, num=0):
    try:
        return leaves_string(hd.head_np_of_np(scope[constituent][num]['TREE'])).rpartition(scope[constituent][num]['HEAD_LIST'][0]['HEAD'])[0].split()
    except:
        return []




def concat_tree_list_to_string(tree_list, sep = ' '):
    tree_string = ''
    for tree in tree_list:
        tree_string += leaves_string(tree) + sep
    return tree_string.strip()

def tree_startswith_labels(tree, labels):
    # if labels = [ [label1, label2], [label3] ], then returns True if the
    # first child of tree is labeled label1 or label2, and the second child is
    # named label3. In general, labels is a list of lists of labels.
    if len(tree) < len(labels):
        return False
    for idx in xrange(len(labels)):
        if tree[idx].node not in labels[idx]:
            return False
    return True

def tree_endswith_labels(tree, labels):
    # if labels = [ [label1, label2], [label3] ], then returns True if the last
    # child of tree is labeled label3, and the second to last child is named
    # label1 or label2. In general, labels is a list of lists of labels.
    if len(tree) < len(labels):
        return False
    for idx in xrange(len(labels)):
        if tree[-idx-1].node not in labels[-idx-1]:
            return False
    return True

def tree_endswith_labels_deep(tree, labels):
    # like tree_endswith_labels, but looks at the deepest rightmost labels.
    # For example, given the tree (NP (NP (DT The) (NN Man)) (PP (IN with) (NP
    # (NN telescope)))). The children of the main NP are labeled NP and PP,
    # which the shallow versions of the similarly named functions look at. But
    # this function looks at, from left-to-right: NP NP DT NN PP IN NP NN
    subtree_list = [subtree for subtree in tree.subtrees()]
    if len(subtree_list) < len(labels):
        return False
    for idx in xrange(len(labels)):
        if subtree_list[-idx-1].node not in labels[-idx-1]:
            return False
    return True


def pp_list_of_np(np_subtree):
    # returns the children PP of an NP
    pp_list = []
    if np_subtree and np_subtree.node == "NP":
        for child in np_subtree:
            if child.node == "PP":
                pp_list.append(child)
    return pp_list

def find_first_constituent(tree, labels):
    # returns the first constituent with a label from the list label, while
    # doing a deep traversal
    for t in tree.subtrees():
        if t.node in labels:
            return t

def prep_of_pp(pp_subtree):
    # returns the prepostion of a PP
    # TODO: does this handle multiple-word prepositions?
    if pp_subtree.node == "PP":
        if len(pp_subtree) > 0 and pp_subtree[0].node in ["IN", "TO"]:
            return pp_subtree[0]

def lists_have_common(treelist1, treelist2):
    # returns True if the two lists share a common element
    for treej in treelist1:
        for treep in treelist2:
            if treej and treej == treep:
                return treej

def find_pre_target_trees(target, tree):
    pretarget_trees = []
    for child in tree.subtrees():

        if not has_subtree(child, target):

            # make sure we have not already looked at a parent of child
            have_looked = False
            for looked in pretarget_trees:
                if has_subtree(looked, child):
                    have_looked = True

            if not have_looked:
                pretarget_trees.append(child)

        if child == target:
            break

    return pretarget_trees

def modifiers_of_head_noun_of_pp_constituent(scope, constituent, num=0):
    return leaves_string(hd.head_np_of_pp(scope[constituent][num]['TREE'])).rpartition(scope[constituent][num]['HEAD_LIST'][0]['HEAD'])[0].split()

def constituent_has_head_stems(scope, constituent, num=0):
    return len(scope[constituent]) > num and len(scope[constituent][num]['HEAD_LIST']) > 0 and len(scope[constituent][num]['HEAD_LIST'][0]['STEMS']) > 0
# given a (PP-OF-XXXX, tree), find scope[XXXX][y]['TREE'] == tree and return (y, XXXX)
def np_constituent_to_which_pp_attaches(scope, pp_key):
    constituent = pp_key[0].split('-')[-1]
    tree = pp_key[1]
    for num, sc in enumerate(scope[constituent]):
        if sc['TREE'] == tree:
            return (num, constituent)

def find_post_target_trees(target, tree):

    posttarget_trees = []
    found_target = False
    for child in tree.subtrees():
        if child == target:
            found_target = True
            posttarget_trees.append(child)
            continue
        if not found_target:
            continue

        have_looked = False
        for looked in posttarget_trees:
            if has_subtree(looked, child):
                have_looked = True
        if not have_looked:
            posttarget_trees.append(child)

    if len(posttarget_trees) > 0:
        return posttarget_trees[1:]
    return posttarget_trees



#def find_post_target_trees(target, tree):
#    posttarget_trees = []
#    subtree_list = [t for t in tree.subtrees()]
#    for child in reversed(subtree_list):
#        if not has_subtree(target, child):
#
#            # make sure we have not already looked at a parent of child
#            have_looked = False
#            for looked in posttarget_trees:
#                if has_subtree(child, looked):
#                    have_looked = True
#
#            if not have_looked:
#                posttarget_trees.append(child)
#
#        if child == target:
#            break
#
#    posttarget_trees.reverse()
#    return posttarget_trees

def tree_split(tree, labels):
    # constituents is returned and will have the form [ [tree1, tree2],
    # [tree3], .. ] where the treei's are children of tree, and they are placed
    # into new sublists whenever a constituent with label in labels appears
    # between them.

    constituents = []
    consti = []
    for child in tree:
        if child.node in labels:
            if len(consti) > 0:
                constituents.append(consti)
            consti = []
        else:
            consti.append(child)

    if len(consti) > 0:
        constituents.append(consti)
    return constituents

def has_subtree(tree, subtree):
    for t in tree.subtrees():
        if t == subtree:
            return True
    return False

def tree_endswith_wordnet_noun(subtree):
    for candidate in hd.head_np_candidate_heads(subtree):
        stemmed = wn.morphy(candidate['HEAD'].lower())
        if stemmed:
            return True
    return False

