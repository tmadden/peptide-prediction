import pace, pace.sklearn
import sklearn.linear_model
import pprint
import numpy


class RidgeAlgorithm(pace.PredictionAlgorithm):
    
    def __init__(self, fmln_m, fmln_n, encoding_name='onehot'):
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_name = encoding_name

    def do_encoding(self, x):
        x = pace.featurization.do_FMLN_encoding(x,
                                                m=self.fmln_m,
                                                n=self.fmln_n)

        return pace.featurization.encode(x, self.encoding_name)


    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)


        encoded_x = self.do_encoding(x)

        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        encoded_x = self.do_encoding(x)

        return self.clf.predict(encoded_x)

numpy.random.seed(31415)
scores = pace.evaluate(lambda: RidgeAlgorithm(5, 3, encoding_name='onehot'), selected_lengths=[8, 9],test_lengths=[8], selected_alleles=['B3501'])
pprint.pprint(scores)
