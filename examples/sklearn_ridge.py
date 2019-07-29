import pace, pace.sklearn
import sklearn.linear_model
import pprint
import numpy


class RidgeAlgorithm(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)

        encoder = pace.sklearn.create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        encoder = pace.sklearn.create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.clf.predict(encoded_x)

numpy.random.seed(31415)
scores = pace.evaluate(RidgeAlgorithm, selected_lengths=[9], selected_alleles=['B3501'])
pprint.pprint(scores)
