import pace, pace.featurization
import random
import pprint

import numpy as np

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init(max_mem_size="2G")  # specify max number of bytes. uses all cores by default.
h2o.remove_all()  # clean slate, in case cluster was already running


class RandomForestLeftRightSplit(pace.PredictionAlgorithm):

    def __init__(self, numTrees, whichAlleleSet):
        self.numTrees = numTrees
        self.whichAlleleSet = whichAlleleSet

        # although these splits work best for unsplit allele sets, they are not the best with splitting
        # cm, cn = pace.featurization.get_allele_sets(whichAlleleSet)

        self.classmembersLEFT, self.canamesLEFT = \
            pace.featurization.build_allele_sets_soft_clustering(whichAlleleSet,12,0.05,'left')
        self.classmembersRIGHT, self.canamesRIGHT = \
            pace.featurization.build_allele_sets_soft_clustering(whichAlleleSet,12,0.05,'right')
        
        print('LEFT SETS :'+str(self.classmembersLEFT))
        print('RIGHT SETS:'+str(self.classmembersRIGHT))
 
    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]

        y = [1] * len(hits) + [0] * len(misses)

        # fmln_x = pace.featurization.do_FMLN_encoding(x)
        # encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        leftx = pace.featurization.do_5d_encoding(pace.featurization.do_LEFT_encoding(x))
        rightx = pace.featurization.do_5d_encoding(pace.featurization.do_RIGHT_encoding(x))

        nxleft = np.array(leftx)
        nxright = np.array(rightx)
        
        ny = np.array(y)

        nxyleft = np.hstack((nxleft, ny[:, None]))
        nxyright = np.hstack((nxright, ny[:, None]))

        xysetsleft, self.alleles_in_this_forest_left = \
            pace.featurization.split_into_sets(nxyleft, xa, self.classmembersLEFT, self.canamesLEFT)
        xysetsright, self.alleles_in_this_forest_right = \
            pace.featurization.split_into_sets(nxyright, xa, self.classmembersRIGHT, self.canamesRIGHT)
        
        # build an RF model for each of these datasets, left
        frame_list_left = []
        self.clf_left = []
        for dset in xysetsleft:
            frame_list_left.append(h2o.H2OFrame(python_obj=dset))
        all_features = frame_list_left[0].col_names[:-1]
        binder_y = frame_list_left[0].col_names[-1]
        for i, df in enumerate(frame_list_left):
            rfmod = H2ORandomForestEstimator(model_id='left' + str(i), ntrees=self.numTrees)
            rfmod.train(x=all_features, y=binder_y, training_frame=df)
            self.clf_left.append(rfmod)

          # build an RF model for each of these datasets, right
        frame_list_right = []
        self.clf_right = []
        for dset in xysetsright:
            frame_list_right.append(h2o.H2OFrame(python_obj=dset))
        all_features = frame_list_right[0].col_names[:-1]
        binder_y = frame_list_right[0].col_names[-1]
        for i, df in enumerate(frame_list_right):
            rfmod = H2ORandomForestEstimator(model_id='right' + str(i), ntrees=self.numTrees)
            rfmod.train(x=all_features, y=binder_y, training_frame=df)
            self.clf_right.append(rfmod)
      

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        leftx = pace.featurization.do_5d_encoding(pace.featurization.do_LEFT_encoding(x))
        rightx = pace.featurization.do_5d_encoding(pace.featurization.do_RIGHT_encoding(x))

        nxleft = np.array(leftx)
        nxright = np.array(rightx)

        xa = [s.allele for s in samples]

        # set up left and right predictions arrays. 
        # initialize with -1 so we can verify at end that all spots got filled
        predylistleft = np.zeros(len(xa)) - 1
        predylistright = np.zeros(len(xa)) - 1

        # get the full allele set:
        ua = list(set(xa))

        # step through these alleles, left version
        for aname in ua:
            print('predicting: working on allele '+aname)
            # get which forests to use for this allele [allele aname]
            # DEBUG here i could have this return a list of forests, instead of indices, directly.
            useforestsleft = pace.featurization.get_all_sets_for_allele(aname,self.alleles_in_this_forest_left, self.canamesLEFT)
            print('which has '+str(len(useforestsleft))+' forests, left')
            useforestsright = pace.featurization.get_all_sets_for_allele(aname,self.alleles_in_this_forest_right, self.canamesRIGHT)
            print('which has '+str(len(useforestsright))+' forests, right')
            # also, find all samples that are allele aname
            myi = np.where(np.in1d(xa, aname))

            # get the x data for these samples.
            nxsetleft = nxleft[myi]
            nxsetright = nxright[myi]

            print(nxleft.shape)
            print(nxright.shape)
            
            # run this data through all of the forests in useforests
            # and gather the responses
            responsesleft=[]
            responsesright=[]
            
            dframe = h2o.H2OFrame(python_obj=nxsetleft)
            for f in useforestsleft:
                pred = self.clf_left[f].predict(dframe)
                plist = pred.as_data_frame().as_matrix().flatten().tolist()
                responsesleft.append(plist)
            r = np.array(responsesleft)
            predylistleft[myi] = r.mean(axis=0)

            dframe = h2o.H2OFrame(python_obj=nxsetright)
            for f in useforestsright:
                pred = self.clf_right[f].predict(dframe)
                plist = pred.as_data_frame().as_matrix().flatten().tolist()
                responsesright.append(plist)
            r = np.array(responsesright)
            predylistright[myi] = r.mean(axis=0)

        # now combine right and left predictions into a single prediction.
        # can use simple element-wise multiplication: idea being that both left and 
        # right need to be binders (1) to be a binder
        # but here use sigmoids instead.
        import math

        def sigmoid(x):
            return 1 / (1 + math.exp(-10*(x-.5)))

        predylist = [sigmoid(pl)*sigmoid(pr) for pl,pr in zip(predylistleft,predylistright)]
        
        return predylist

whichAlleleSet = 95

my_scorers = {'ppv': pace.evaluation.PpvScorer(), 'accuracy': pace.evaluation.AccuracyScorer(cutoff=0.6)}

test_alleles = pace.featurization.a16_names

scores = pace.evaluate(lambda : RandomForestLeftRightSplit(20, whichAlleleSet), scorers=my_scorers,
                       selected_lengths=[8,9,10,11],dataset=pace.data.load_dataset(whichAlleleSet),
                       test_alleles=test_alleles)
pprint.pprint(scores)
# print averages too
# note we should rename accuracy scorer to "weighted_accuracy"
print(sum(scores['accuracy'])/len(scores['accuracy']))
print(sum(scores['ppv'])/len(scores['ppv']))
