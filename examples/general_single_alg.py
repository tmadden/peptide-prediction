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
        x = pace.featurization.do_FMLN_encoding(x,
                                                m=self.fmln_m,
                                                n=self.fmln_n)

        return pace.featurization.encode(x, self.encoding_name)


class isolated_alg(pace.PredictionAlgorithm, encoder_super):

    def __init__(self, fmln_m, fmln_n, encoding_name='onehot', alg_name='ridge'): 
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_name = encoding_name
        self.alg_name = alg_name

    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        encoded_x = self.do_encoding(x)

        if self.alg_name == 'ridge':
            self.clf = sklearn.linear_model.RidgeClassifier()
        elif self.alg_name == 'randomforest':
            self.clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=np.random.seed(31415))
        elif self.alg_name == 'svmlin': 
            self.clf = sklearn.svm.SVC(C=1, kernel='linear', probability=True)
        elif self.alg_name == 'svmrbf': 
            self.clf = sklearn.svm.SVC(C=10, kernel='rbf', gamma='scale', probability=True)

        self.clf.fit(encoded_x, y)
        
        #below are a few examples which use cross validation loop to optimization hyperparameters.
        #leaving them commented out for now but they are there if we need them.

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

        #Cs = 10 means 10 values chosen on log scale between 1e-4 and 1e4
        #default for logisticRegressionCV is L2 regularization --> ridge.
        #one problem with this method is we can't see the winning hyperparameter chosen :(
        #self.clf = LogisticRegressionCV(Cs=10, cv=5).fit(encoded_x, y)

        #instead use logregression with gridsearchcv so i can see optimal hyperparams:
        '''
        param_grid = {'C': [.1, .5, 1, 10, 50, 100]}
        self.clf = GridSearchCV(
            LogisticRegression(solver='lbfgs'), param_grid, cv=5)
        print('executing grid search...')
        self.clf.fit(encoded_x, y)
        print("Best parameters found:")
        print(self.clf.best_params_)
        '''

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = self.do_encoding(x)
        if self.alg_name == 'ridge':
            return self.clf.predict(encoded_x)
        else:
            r = self.clf.predict_proba(encoded_x)
            return r[:, 1]

np.random.seed(31415)
scores = pace.evaluate(lambda: isolated_alg(5, 3, encoding_name='onehot', 
    alg_name='svmrbf'), selected_lengths=[9],test_lengths=[9], selected_alleles=['B3501'])
pprint.pprint(scores)