import pace, pace.sklearn, pace.featurization
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import pprint


class encoder_super():
    def do_encoding(self, x):
        x = pace.featurization.do_FMLN_encoding(
            x, m=self.fmln_m, n=self.fmln_n)
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


class isolated_rf(pace.PredictionAlgorithm, encoder_super):
    def __init__(self, fmln_m, fmln_n, encoding_style='one_hot'):
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_style = encoding_style

    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        encoded_x = self.do_encoding(x)

        self.clf = RandomForestClassifier(
            n_estimators=30, max_depth=None, random_state=np.random.seed(1234))

        self.clf.fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = self.do_encoding(x)
        r = self.clf.predict_proba(encoded_x)
        return r[:, 1]


class isolated_alg(pace.PredictionAlgorithm, encoder_super):
    def __init__(self, fmln_m, fmln_n, encoding_style='one_hot'):
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_style = encoding_style

    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        encoded_x = self.do_encoding(x)
        '''
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=np.random.seed(1234))
        '''

        #self.clf = sklearn.svm.SVC(C=10, kernel='rbf', gamma='scale', probability=True)
        #self.clf = sklearn.svm.SVC(C=.1, kernel='linear', probability=True)

        #example with hyperparam cross val search:
        #to do: try using ppv scorer also:
        #usescore = make_scorer(pace.evaluation.score_by_ppv)
        '''
        param_grid = {'C': [.05, .1, .2]}
        self.clf = GridSearchCV(
            sklearn.svm.SVC(kernel='linear', probability=True),
            param_grid,
            cv=5)
        print('executing grid search...')
        self.clf.fit(encoded_x, y)
        print("Best parameters found:")
        print(self.clf.best_params_)
        '''

        #self.clf.fit(encoded_x, y)

        #Cs = 10 means 10 values chosen on log scale between 1e-4 and 1e4
        #default for logisticRegressionCV is L2 regularization --> ridge.
        #one problem with this method is we can't see the winning hyperparamter chosen :(
        #self.clf = LogisticRegressionCV(Cs=10, cv=5).fit(encoded_x, y)

        #instead use logregression with gridsearchcv so i can see optimal hyperparams:
        param_grid = {'C': [.1, .5, 1, 10, 50, 100]}
        self.clf = GridSearchCV(
            LogisticRegression(solver='lbfgs'), param_grid, cv=5)
        print('executing grid search...')
        self.clf.fit(encoded_x, y)
        print("Best parameters found:")
        print(self.clf.best_params_)

        #self.clf.fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = self.do_encoding(x)
        r = self.clf.predict_proba(encoded_x)
        return r[:, 1]
