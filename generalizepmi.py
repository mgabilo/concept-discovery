#!/usr/bin/python
import features
import glob
from nltk.corpus import wordnet as nlwn
import sys
import pdb
import operator

class ConceptNode(object):
    def __init__(self, name, value=0, max=0):
        self.value = value
        self.max = max
        self.name = name

def degrade_subconcept_ic(subconcept_val, subconcept, superconcept):
    superconcept_ic = features.information_content(superconcept, features.ic)
    subconcept_ic = features.information_content(subconcept, features.ic)
    
    if superconcept_ic == 0 or subconcept_ic == 0:
        if superconcept.name != 'entity.n.01':
            shared_ic_ratio = 1.0
        else:
            shared_ic_ratio = 0.0
    else:
        shared_ic_ratio = superconcept_ic / subconcept_ic
    degraded = subconcept_val * shared_ic_ratio
    if degraded <= subconcept_val:
        return degraded
    return subconcept_val


def generalize_and_save_pmi(source_dir, dest_dir):
    for pmi_filename in glob.glob('%s/*.pmi' % source_dir):
        pmi_model = features.PMINounModel()
        pmi_model._load_from_pmi_file(pmi_filename)
        generalize_pmi_nounmodel(pmi_model)
        pmi_model.save_to_file(dest_dir)


def generalize_pmi_nounmodel(pmi_noun_model):
    concept_weights_dict = {}

    for section in pmi_noun_model.sections:
        if section in ['subject_head_of_verbs', 'object1_head_of_verbs', 'object2_head_of_verbs', 'modified_heads', 'coheads']:

            lemma_pmi_set = []
            for lemma in pmi_noun_model.__dict__[section]:
                pmi = pmi_noun_model.__dict__[section][lemma]
                lemma_pmi_set.append( (lemma, pmi) )
            if section in ['modified_heads', 'coheads']:
                concept_weights = get_concept_weights_for_lemma_pmi_set(lemma_pmi_set, 'n')
            else:
                concept_weights = get_concept_weights_for_lemma_pmi_set(lemma_pmi_set, 'v')

            if concept_weights:
                concept_weights_dict[section] = concept_weights

        if section in ['prep_head_noun_of_pp_pairs', 'prep_head_noun_pairs', 'prep_verb_pairs', 'modifiers_of_head']:
            lemma_pmi_sets = {}
            for prep_lemma in pmi_noun_model.__dict__[section]:
                try:
                    prep, lemma = prep_lemma
                    pmi = pmi_noun_model.__dict__[section][(prep, lemma)]
                    if not lemma_pmi_sets.has_key( (section, prep) ):
                        lemma_pmi_sets[ (section, prep) ] = []
                    lemma_pmi_sets[ (section, prep) ].append((lemma, pmi))
                except Exception, e:
                    print 'error adding', prep_lemma, 'from', section
                    print e

            for section_prep in lemma_pmi_sets:
                section_prep_lemmas = lemma_pmi_sets[section_prep]
                if section in ['prep_verb_pairs']:
                    concept_weights = get_concept_weights_for_lemma_pmi_set(section_prep_lemmas, 'v')
                elif section in ['prep_head_noun_of_pp_pairs', 'prep_head_noun_pairs']:
                    concept_weights = get_concept_weights_for_lemma_pmi_set(section_prep_lemmas, 'n')

                else:
                    # modifiers_of_head
                    section, pos = section_prep # e.g. ('ADJ', 'big')
                    if pos == 'ADJ':
                        pos = 'a'
                    elif pos == 'N':
                        pos = 'n'
                    concept_weights = get_concept_weights_for_lemma_pmi_set(section_prep_lemmas, pos)

                concept_weights_dict[section_prep] = concept_weights
        pmi_noun_model.__dict__[section] = {}

    for section_prep in concept_weights_dict:
        for concept in concept_weights_dict[section_prep]:
            concept_weight = concept_weights_dict[section_prep][concept].value

            if isinstance(section_prep, tuple):
                section, prep = section_prep
                if section == 'modifiers_of_head':
                    pos = prep
                    if pos == 'n':
                        pos = 'N'
                    elif pos == 'a':
                        pos = 'ADJ'
                    pmi_noun_model.__dict__[section][ (pos, concept) ] = concept_weight
                else:
                    pmi_noun_model.__dict__[section][ (prep, concept) ] = concept_weight
            else:
                section = section_prep
                pmi_noun_model.__dict__[section][ concept ] = concept_weight


def get_concept_weights_for_lemma_pmi_set(lemma_pmi_set, pos):
    concept_weights = {}

    if pos in ['n', 'v']:
        for lemma, pmi in lemma_pmi_set:
            add_word_to_concept_weights(lemma, pmi, concept_weights, pos)

    elif pos == 'a':
        for lemma, pmi in lemma_pmi_set:
            add_adj_to_adj_weights(lemma, pmi, concept_weights)
    return concept_weights



def add_word_to_concept_weights(word, word_pmi, concept_nodes, pos):
    synsets = nlwn.synsets(word, pos)
    if not synsets:
        return

    synset_weight = (1.0 / len(synsets))
    sorted_concept_weights = []

    for synset in synsets:
        concept_weight = word_pmi * synset_weight
        if concept_weight > 0.0:
            sorted_concept_weights.append( (synset, concept_weight) )
    sorted_concept_weights.sort(key=operator.itemgetter(1), reverse=True)

    for synset, weight in sorted_concept_weights:
        if not concept_nodes.has_key(synset.name):
            concept_nodes[synset.name] = ConceptNode(synset.name)
        update_concept_and_superconcepts(concept_nodes, concept_nodes[synset.name], weight, weight)


def add_adj_to_adj_weights(adj, word_pmi, concept_nodes):
    synsets = nlwn.synsets(adj, 'a')
    if not synsets:
        return

    synset_weight = (1.0 / len(synsets))
    sorted_concept_weights = []

    for synset in synsets:
        concept_weight = word_pmi * synset_weight
        if concept_weight > 0.0:
            sorted_concept_weights.append( (synset, concept_weight) )
    sorted_concept_weights.sort(key=operator.itemgetter(1), reverse=True)

    for synset, weight in sorted_concept_weights:
        if not concept_nodes.has_key(synset.name):
            concept_nodes[synset.name] = ConceptNode(synset.name)
        update_similarto_weights(concept_nodes, concept_nodes[synset.name], weight, word_pmi)


def update_similarto_weights(concept_nodes_dict, concept_node, value_new, max_new, level=0, decay=0.5, max_level=3):

    concept_node.value = value_new
    concept_node.max = max(max_new, concept_node.max)
    parent_max_new = concept_node.max

    if level == max_level:
        return

    synset = nlwn.synset(concept_node.name)
    for superconcept in synset.similar_tos():
        concept_node_value_icloss = concept_node.value * decay
        if not concept_nodes_dict.has_key(superconcept.name):
            concept_nodes_dict[superconcept.name] = ConceptNode(superconcept.name)
            update_similarto_weights(concept_nodes_dict, concept_nodes_dict[superconcept.name], \
                    concept_node_value_icloss, parent_max_new, level+1)

        else:
            superconcept_node = concept_nodes_dict[superconcept.name]
            concept_max = max(concept_node.max, superconcept_node.max) * 2
            learn_rate = (concept_node_value_icloss + superconcept_node.value) / concept_max
            if learn_rate > 1.0: pdb.set_trace()
            parent_scale = superconcept_node.max - superconcept_node.value
            parent_value_new = learn_rate * parent_scale + superconcept_node.value
            update_similarto_weights(concept_nodes_dict, superconcept_node, parent_value_new, parent_max_new, level+1)




def update_concept_and_superconcepts(concept_nodes_dict, concept_node, value_new, max_new):

    concept_node.value = value_new
    concept_node.max = max(max_new, concept_node.max)


    parent_max_new = concept_node.max

    synset = nlwn.synset(concept_node.name)
    for superconcept in synset.hypernyms():
        concept_node_value_icloss = degrade_subconcept_ic(concept_node.value, synset, superconcept)
        if not concept_nodes_dict.has_key(superconcept.name):
            concept_nodes_dict[superconcept.name] = ConceptNode(superconcept.name)
            update_concept_and_superconcepts(concept_nodes_dict, concept_nodes_dict[superconcept.name], \
                    concept_node_value_icloss, parent_max_new)

        else:
            superconcept_node = concept_nodes_dict[superconcept.name]

            concept_max = max(concept_node.max, superconcept_node.max) * 2
            learn_rate = (concept_node_value_icloss + superconcept_node.value) / concept_max
            if learn_rate > 1.0: pdb.set_trace()
            parent_scale = superconcept_node.max - superconcept_node.value
            parent_value_new = learn_rate * parent_scale + superconcept_node.value
            update_concept_and_superconcepts(concept_nodes_dict, superconcept_node, parent_value_new, parent_max_new)



def main():
    if len(sys.argv) != 3:
        print 'Arguments: <source pmi dir> <dest pmi dir>'
        return

    source_pmi_dir = sys.argv[1]
    dest_pmi_dir = sys.argv[2]
    generalize_and_save_pmi(source_pmi_dir, dest_pmi_dir)

if __name__ == "__main__":
    main()

