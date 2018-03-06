# concept-discovery

The full description of the system can be found in the paper [on
this page](http://mgabilo.org/concept-discovery.html).

Here is a modified excerpt from the abstract:

This program discovers additional ontological categories for words
which are underrepresented in the WordNet ontology.  The basic
approach begins with extracting local syntactic dependencies for each
word by parsing a large corpus to construct feature vectors for those
words.  The words' feature vectors are used to construct feature
vectors for upper-level ontological concepts by bootstrapping from the
existing WordNet ontology.  A similarity metric between these two
types of feature vectors is used to discover ontological categories
for some words.


## Download the dataset 

These are the frequency-based feature vectors for each word that
resulted from parsing about 1 million sentences of Wikipedia. Choose
from one of the following mirrors.

* [Mirror 1 - Google drive](https://drive.google.com/file/d/1GaBpQaHZOny-rVPc3lxgL_1Zxq2UFVRY/view?usp=sharing)
* [Mirror 2 - Dropbox](https://www.dropbox.com/s/jfzb78ivc6jwdbi/word-feature-frequencies.tar?dl=1)
* [Mirror 3 - MEGA](https://mega.nz/#!yjwzXYaZ!wGAIkm3VvNrE3jPtDSC6BwmyR9ldEi6dfSPPCf_M2-c)


## Requirements

This project requires Python 2.7 (not 3.x) and NLTK for Python
2.x. The WordNet 3.0 corpus for NTLK is required. The binary
/usr/bin/python should point to Python 2.7. In Ubuntu 16.04 the
command would be the following.


```
apt-get install python2.7 python-nltk
```

To download the WordNet 3.0 corpus from NLTK run the following and use
the resulting GUI to download the corpus.


```
>>> import nltk
>>> nltk.download()
```


## Running the program

This repository already includes the final results of running the
steps in this section as the following files:

* [concept-vectors-upper-level](concept-vectors-upper-level) (directory)

* [evaluation/word-concept-topsim.txt](evaluation/word-concept-topsim.txt)

* [evaluation/newly-discovered-word-concept.txt](evaluation/newly-discovered-word-concept.txt)



Download and untar the dataset given in the section above. Move the
resulting directory "word-feature-frequencies" into the base directory
of this repository. I'll assume the current/working directory is the
base directory of this repository.

Generate the PMI-based word feature vectors from the frequency-based
word feature vectors. These are also more human-readable than the
files in word-feature-frequencies, so you can easily inspect them.

```
mkdir word-feature-pmi
./features.py -savepmi word-feature-frequencies word-feature-pmi
```

Create the index files for each word and feature in word-feature-pmi,
and the index file for each synset (concept) in WordNet.

```
./index_features.py word-feature-pmi features.idx
./index_words.py word-feature-pmi words.idx
cp words.idx words-to-be-processed
./index_concepts.py concepts.idx
```

Build an SQLite DB from the files in word-feature-pmi.

```
./make_word_db.py word-feature-pmi word-feature-pmi.db
```


Generate the file that contains the instructions to generate concept
vectors. This file specifies the pairs of word vectors which should be
intersected together to create concept vectors.

```
./generate-cv-script.py cv-script
```

Generate the concept vectors.

```
mkdir cv
./generate_cv.py cv-script word-feature-pmi cv
```


Generate a set of concept vectors where each concept vector contains
the union of all its descendants and itself. These are only used in a
similarity metric for building the upper-level concept vectors in the
step after this one.

```
mkdir cv-clustered
./build_clustered_cv.py cv cv-clustered
```


Build the set of upper-level concept vectors which will be used to
categorize words, and build the corresponding SQLite DB.


```
mkdir cv-upper-level
./prune_concepts.py cv cv-clustered cv-upper-level
./make_pmi_db.py cv-upper-level cv-upper-level.db
```


Categorize words under the upper-level concepts. This is currently set
up to only load words containing more than 100 features with a value
of 7.0 (PMI) or more. This can be tweaked in the function
make_pmi_db.py:load_words_mem_db.


```
./word_concept_topsim.py word-feature-pmi.db cv-upper-level.db > word-concept-topsim.txt
```


View only the word-concept categorizations that are novel, i.e., not
already in WordNet.  The script also takes some measures to avoid
proper names being categorized since they were problematic for the
judges to evaluate.

```
./check_word_concept_topsim.py word-concept-topsim.txt > newly-discovered-word-concept.txt
```


## Evaluation

The steps above do not reproduce the same results [documented in the
paper](http://mgabilo.org/concept-discovery.pdf).  However, I've
performed a new evaluation for the current data.  The results of the
evaluation can be found in the following files.

* [evaluation/judge1.txt](evaluation/judge1.txt)
* [evaluation/judge2.txt](evaluation/judge2.txt)
* [evaluation/judge1-judge2-agreement.txt](evaluation/judge1-judge2-agreement.txt)

The two judges were given the same 200 random word-concept pairs from
the file newly-discovered-word-concept.txt.  The task was to judge
whether the word has a sense subsumed by the concept. Judges 1 and 2
respectively determined 63% and 72% of the word-concept pairs to be
correct.  The judges agreed with each other for 152 of the pairs,
yielding an inter-annotator agreement (IAD) of 76% (i.e., given a
word-concept pair, they both either said "Correct" or "Wrong").  The
two judges then discussed the 48 pairs on which they disagreed to come
to an agreement, which resulted in a shared agreement of 63.5% of the
word-concept pairs being correct.

## What's not included

The software used to parse sentences in Wikipedia was a modified
version of the Charniak parser which minimally reconstructed clause
structures from the parse tree. The features (frequency-based word
vectors) were then extracted from these minimal clause structures.
This software is not currently available due to unclear licensing
restrictions. In addition, the parses of the Wikipedia sentences are
also not available.


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE) file for details.

## Authors

* **Michael Gabilondo** - [mgabilo](https://github.com/mgabilo)


