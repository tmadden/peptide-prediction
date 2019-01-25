import pace, pace.featurization
import random
import pprint

import numpy as np

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator


h2o.init(max_mem_size="2G")  # specify max number of bytes. uses all cores by default.
h2o.remove_all()  # clean slate, in case cluster was already running


class RandomForesth2oWithoutAlleles(pace.PredictionAlgorithm):

    def __init__(self, nt):
        self.numTrees = nt

    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]

        y = [1] * len(hits) + [0] * len(misses)

        # for 9 mers only, FMLN encoding with 5,4 does nothing, so can compare easily with other '9 only' algs.
        fmln_x = pace.featurization.do_FMLN_encoding(x,m=5,n=4)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        nx1 = np.array(encoded_x)
        ny = np.array(y)
        nxy = np.hstack((nx1, ny[:, None]))
        myh2oframe = h2o.H2OFrame(python_obj=nxy)

        # set which columns for x, which for y
        all_features = myh2oframe.col_names[:-1]
        binder_y = myh2oframe.col_names[-1]
        print('numTrees='+str(self.numTrees))
        self.clf = H2ORandomForestEstimator(
            model_id="v2",
            ntrees=self.numTrees)
        # stopping_rounds=2,
        # score_each_iteration=True,
        # seed=10)

        self.clf.train(x=all_features, y=binder_y, training_frame=myh2oframe)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        fmln_x = pace.featurization.do_FMLN_encoding(x,m=5,n=4)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        nx1 = np.array(encoded_x)

        myh2oframe = h2o.H2OFrame(python_obj=nx1)

        pred = self.clf.predict(myh2oframe)
        # h2o gives h2oframe result: convert to python list:
        predylist = pred.as_data_frame().as_matrix().flatten().tolist()
        return predylist

# just a subset of the lengths:
p = [9]
# or all:
# p = [8, 9, 10, 11]

scores = pace.evaluate(lambda : RandomForesth2oWithoutAlleles(55),
                       **pace.load_data_set(16, peptide_lengths=p))
pprint.pprint(scores)
