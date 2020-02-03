import pace, pace.sklearn, pace.featurization
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sklearn.linear_model
import pprint


class voting_fmln(pace.PredictionAlgorithm):
    def __init__(self, fmln_m, fmln_n, encoding_style='one_hot'):
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_style = encoding_style

    def do_encoding(self, x):
        x = pace.featurization.do_FMLN_encoding(x,
                                                m=self.fmln_m,
                                                n=self.fmln_n)
        if self.encoding_style == 'one_hot':
            encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
            encoder.fit(x)
            return encoder.transform(x).toarray()
        elif self.encoding_style == '5d':
            return pace.featurization.do_5d_encoding(x)
        elif self.encoding_style == 'both':
            encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
            encoder.fit(x)
            ronehot = encoder.transform(x).toarray()
            r5d = pace.featurization.do_5d_encoding(x)
            return np.concatenate((ronehot, r5d), axis=1)
        else:
            raise Exception('Unknown encoding style used: ' +
                            self.encoding_style)

    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        encoded_x = self.do_encoding(x)

        self.clf = VotingClassifier(
            estimators=[
                ('svmrbf',
                 sklearn.svm.SVC(C=10,
                                 kernel='rbf',
                                 gamma='scale',
                                 probability=True)),
                ('svmlin',
                 sklearn.svm.SVC(C=.1, kernel='linear', probability=True)),
                ('rf',
                 RandomForestClassifier(n_estimators=30,
                                        max_depth=None,
                                        random_state=np.random.seed(1234))
                 )  #for reproduceable results set random_state. note sklearn uses numpy.random, pace splitting uses random
            ],
            voting='soft')
        self.clf.fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = self.do_encoding(x)
        r = self.clf.predict_proba(encoded_x)
        return r[:, 1]

import logging
logging.basicConfig(level=logging.INFO)

#shortcuts:
a02 = ['A0203']
a68 = ['A6802']
b35 = ['B3501']
#my_scorers = {'ppv': pace.evaluation.PpvScorer(), 'accuracy': pace.evaluation.AccuracyScorer(cutoff=0.6)}

#single call:

scores = pace.evaluate(lambda: voting_fmln(5, 4, encoding_style='one_hot'),
                       selected_lengths=[9],
                       selected_alleles=b35,
                       dataset=pace.data.load_dataset(16),
                       nbr_train=1,
                       nbr_test=10,
                       random_seed=1)
#note: pinot noir values for nbr_train and nbr_test: 10, 1000
pprint.pprint(scores)


