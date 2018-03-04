** word-feature-frequencies.tar download mirrors:

https://drive.google.com/file/d/1GaBpQaHZOny-rVPc3lxgL_1Zxq2UFVRY/view?usp=sharing

https://www.dropbox.com/s/jfzb78ivc6jwdbi/word-feature-frequencies.tar?dl=1

https://mega.nz/#!yjwzXYaZ!wGAIkm3VvNrE3jPtDSC6BwmyR9ldEi6dfSPPCf_M2-c




# parsing

---


# creating word vectors from parses
## makes freq files from parses and writes to word-feature-frequencies
## we use pre-existing one since we don't have the parses, also
## these files were "post processed due to bugs in counting"
### ./features.py -count word-feature-frequencies

## generates pmi files from freq files and writes to pmi-dir
mkdir pmi-dir

## IMPORTANT: below script can ignore small/negative PMIs or not. the
   original wc-filter directory did contain negative PMIs but the
   script had a line that ignored them. currently we are including
   negative PMIs. (around line 726: if float(pmi) < 0.01: continue)
   --- it seems above note is irrelevant because make_word_db.py
       ignores PMI less than 3.0 and any features with < 40 times

$ ./features.py -savepmi word-feature-frequencies word-feature-pmi

## assigns an index to each feature of the words in directory
   word-feature-pmi -- outputs to features.idx

$ ./index_features.py word-feature-pmi features.idx


## assigns an index to each word of the words in directory
   word-feature-pmi -- outputs to words.idx

$ ./index_words.py word-feature-pmi words.idx
$ cp words.idx words-to-be-processed

## assigns an index to each word in wordnet -- outputs to
   concepts.idx
./index_concepts.py concepts.idx


## builds an sqlite db from the files in word-feature-pmi; reads
   features.idx and words.idx for indices

$ ./make_word_db.py word-feature-pmi word-feature-pmi.db





# creating concept vectors from word vectors

## note: the original seems to have been cv-script-nomono and it's
##  slightly different from new-cv-script

## possible explanation: hyponyms_by_level does "if not
   similar_enough: break" if one of concepts of the previous level has
   a hyponym that is not similar enough to the concept. however, this
   check is done for each concept in the previous level in order, and
   if one of those concepts passes the check, the hyponyms are added
   to the next level. it's possible an older python implementation
   returned hyponyms in a different order than the current one,
   leading to different behavior.

## I have modified generate-cv-script.py so that the order of the
   concepts returned does not matter -- now the generated cv-script is
   much larger since it's not breaking prematurely on concepts that
   have no hyponyms (which was another problem that I fixed)
   -- hyponyms_by_level & new-cv-script & new-cv-nomono


## note: "nomono" means word vectors of monosemous words by themselves
##   are not unioned with the concept vectors; doing this was an early
##   idea but it was scrapped later;
## instead, any word vector MUST be intersected with another word vector
##    in order to be unioned with the concept vectors (same as described
##    in the paper)

./generate-cv-script.py new-cv-script

## note: the script below loads the word vectors (*.pmi.bz2) and
##   creates concept vectors; however, the features called
##   "modifiers_of_head" and "coheads" are not loaded (they aren't used)
## ("modifiers_of_head" is when the word is the head noun and we
##   record the preceding modifiers); this is consistent with the paper

mkdir new-cv-nomono
./generate_cv.py new-cv-script word-feature-pmi new-cv-nomono


############################

mkdir new-cv-nomono-clusters
./build_clustered_cv.py new-cv-nomono new-cv-nomono-clusters

mkdir new-cv-nomono-clusters-pruned-0.75-100
./prune_concepts.py new-cv-nomono new-cv-nomono-clusters new-cv-nomono-clusters-pruned-0.75-100

./make_pmi_db.py new-cv-nomono-clusters-pruned-0.75-100 new-cv-nomono-clusters-pruned-0.75-100.db


# make_pmi_db.py:load_words_mem_db -- loads only words with > 100 features with 7.0 pmi or more

./word_concept_topsim2.py word-feature-pmi.db new-cv-nomono-clusters-pruned-0.75-100.db > word-concept-topsim-new-100-feature-words


# make_pmi_db.py:load_words_mem_db -- loads only words with > 30 features with 7.0 pmi or more

./word_concept_topsim2.py word-feature-pmi.db new-cv-nomono-clusters-pruned-0.75-100.db > word-concept-topsim-new-30-feature-words


./check_word_concept_topsim.py word-concept-topsim-new-30-feature-words > newly-discovered-30-feature-words

./check_word_concept_topsim.py word-concept-topsim-new-100-feature-words > newly-discovered-100-feature-words-no-instances


shuf newly-discovered-100-feature-words-no-instances | head -n200 > random-200.txt


############################



# categorizing words under committees (clusters/unions of concept vectors)

$ ./word_concept_topsim2.py words-idx.db cv-upper-level-no-parent-0.65-200-noparentfix.db
loaded # words: 28242
loaded # concepts: 393
["nunnery", [["house.n.01", 0.18180907590327747], ["community.n.01", 0.07075934504162712], ["site.n.01", 0.06348653254310868], ["real_property.n.01", 0.0597139386462756]]]
["poplar", [["plant.n.02", 0.08614739247360066], ["community.n.06", 0.05789244611677778], ["natural_depression.n.01", 0.056951500523193276], ["body_of_water.n.01", 0.047868515922207276]]]
["circuitry", [["structure.n.04", 0.08667986870944122], ["electronic_equipment.n.01", 0.08356154509877362], ["shape.n.02", 0.06989125193371416], ["care.n.01", 0.06944638221151396], ["arrangement.n.02", 0.06810696284845308], ["know-how.n.01", 0.06529706494917999], ["part.n.02", 0.06118626004691826], ["organic_process.n.01", 0.06011953298862135], ["writing.n.04", 0.05847264144039635], ["plant_part.n.01", 0.055716188916546665], ["controlled_substance.n.01", 0.05501659374350455], ["part.n.01", 0.05464198525396642], ["system_of_measurement.n.01", 0.05017361355124943], ["motivation.n.01", 0.0491852932536199], ["plant.n.02", 0.04678034801573678], ["topographic_point.n.01", 0.04528635354164434]]]
["cycling", [["sport.n.01", 0.24063101378469792], ["wit.n.01", 0.10099212924009421], ["knowledge_domain.n.01", 0.09802943917047569], ["medium.n.01", 0.09496991052927228], ["traveler.n.01", 0.08741126855571346], ["organic_process.n.01", 0.0824757328426007], ["carrier.n.05", 0.07940260509512688], ["substance.n.07", 0.07697597019799536], ["definite_quantity.n.01", 0.07110696285293623], ["trait.n.01", 0.06389338300251632], ["part.n.01", 0.0618120573605581], ["odd-toed_ungulate.n.01", 0.0547736368767819], ["natural_depression.n.01", 0.05385022674357195], ["point.n.02", 0.05322724081654276], ["part.n.02", 0.04757771282067214]]]
["cycling", [["change.n.03", 0.0556646556382284], ["traveler.n.01", 0.0548567283550374], ["organic_process.n.01", 0.054562729485230034], ["trait.n.01", 0.046450415084719136]]]

# output above does not match file below -- different concepts/scores

$ cat word-concept-topsim-no-parent-0.65-200-noparentfix 

["nunnery", [["house.n.01", 0.16713087082742689], ["community.n.01", 0.067121903188820886], ["real_property.n.01", 0.065518965002784577], ["site.n.01", 0.059802812060792992], ["aristocrat.n.01", 0.047976175102223254]]]
["poplar", [["plant.n.02", 0.076723387787163852], ["natural_depression.n.01", 0.051264960653516475], ["community.n.06", 0.047724487787159736]]]
["circuitry", [["device.n.01", 0.097700758829796758], ["structure.n.04", 0.083077900250169021], ["faculty.n.01", 0.0819737296079706], ["part.n.02", 0.075925021653584987], ["writing.n.04", 0.075584858409022015], ["arrangement.n.02", 0.068222907935206203], ["communication.n.01", 0.067348445526153419], ["property.n.02", 0.058816368171752495], ["part.n.01", 0.0572351590819122], ["system_of_measurement.n.01", 0.051135485319993595], ["organic_process.n.01", 0.050111754135730283], ["controlled_substance.n.01", 0.045714127738044785]]]
["cycling", [["sport.n.01", 0.21737266171211245], ["knowledge_domain.n.01", 0.084787298556896054], ["traveler.n.01", 0.084777967678478741], ["container.n.01", 0.083022311874975135], ["organic_process.n.01", 0.075388839453071024], ["wit.n.01", 0.075032894202420305], ["carrier.n.05", 0.069268521285439144], ["substance.n.07", 0.065937573093351409], ["state.n.02", 0.057881699939497253], ["definite_quantity.n.01", 0.056558015460630121], ["part.n.01", 0.053381131038613354], ["natural_elevation.n.01", 0.050390774763748807], ["odd-toed_ungulate.n.01", 0.048117807264367568], ["part.n.02", 0.04515769002324331]]]
["cycling", [["organic_process.n.01", 0.052344860861868472], ["change.n.03", 0.051649086255805707]]]



$ ./check_word_concept_topsim.py word-concept-topsim-no-parent-0.65-200-noparentfix

["writings", [["written_communication.n.01", 0.22257636689431556], ["belief.n.01", 0.15210446788675425], ["work.n.02", 0.1354580542582977]]]
["nunnery", [["house.n.01", 0.1671308708274269]]]
["yellow", [["visual_property.n.01", 0.2628939015998629]]]
["narcotic", [["agent.n.03", 0.14840931292888376]]]











