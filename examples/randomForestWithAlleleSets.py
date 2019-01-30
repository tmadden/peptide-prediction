import pace, pace.featurization
import random
import pprint

import numpy as np

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init(max_mem_size="2G")  # specify max number of bytes. uses all cores by default.
h2o.remove_all()  # clean slate, in case cluster was already running


class RandomForestWithAlleleSets(pace.PredictionAlgorithm):

    def __init__(self, numTrees, whichAlleleSet):
        self.numTrees = numTrees
        self.whichAlleleSet = whichAlleleSet
        self.classmembers, self.canames = pace.featurization.get_allele_sets(whichAlleleSet)


    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]

        y = [1] * len(hits) + [0] * len(misses)

        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        nx1 = np.array(encoded_x)
        ny = np.array(y)

        nxy = np.hstack((nx1, ny[:, None]))

        xysets, self.alleles_in_this_forest = pace.featurization.split_into_sets(nxy, xa, self.classmembers, self.canames)
        # alleles_in_this_forest: integer allele numbers. same data as classmembers, but if 
        # there were no xa data in some particular subset, we don't return that set.
        
        # build an RF model for each of these datasets:
        frame_list = []
        self.clf = []

        for dset in xysets:
            frame_list.append(h2o.H2OFrame(python_obj=dset))

        # set which columns for x, which for y: same for all sub sets
        all_features = frame_list[0].col_names[:-1]
        binder_y = frame_list[0].col_names[-1]

        # train all the separate models:

        for i, df in enumerate(frame_list):
            rfmod = H2ORandomForestEstimator(model_id='model' + str(i), ntrees=self.numTrees)
            rfmod.train(x=all_features, y=binder_y, training_frame=df)
            self.clf.append(rfmod)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)
        nx = np.array(encoded_x)

        xa = [s.allele for s in samples]

        # set up my predictions array. initialize with -1 so we can verify at end that all spots got filled
        predylist = np.zeros(len(xa)) - 1

        # get the full allele set:
        ua = list(set(xa))

        # step through these alleles

        for aname in ua:
            print('working on allele '+aname)
            # get which forests to use for this allele [allele aname]
            # DEBUG here i could have this return a list of forests, instead of indices, directly.
            useforests = pace.featurization.get_all_sets_for_allele(aname,self.alleles_in_this_forest, self.canames)
            print('which has '+str(len(useforests))+' forests')
            # also, find all samples that are allele aname
            myi = np.where(np.in1d(xa, aname))

            # get the x data for these samples.
            nxset = nx[myi]
            print(nx.shape)

            # run this data through all of the forests in useforests
            # and gather the responses
            responses=[]
            dframe = h2o.H2OFrame(python_obj=nxset)
            for f in useforests:
                pred = self.clf[f].predict(dframe)
                plist = pred.as_data_frame().as_matrix().flatten().tolist()
                responses.append(plist)
            r = np.array(responses)
            predylist[myi] = r.mean(axis=0)

        return predylist

whichAlleleSet = 16

scores = pace.evaluate(lambda : RandomForestWithAlleleSets(20, whichAlleleSet),
                       **pace.load_data_set(whichAlleleSet, peptide_lengths=[8, 9, 10, 11]))
pprint.pprint(scores)
