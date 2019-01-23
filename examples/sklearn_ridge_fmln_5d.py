import pace, pace.sklearn, pace.featurization
import sklearn.linear_model
import pprint


class RidgeAlgorithmFMLN5D(pace.PredictionAlgorithm):
    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        y = [1] * len(hits) + [0] * len(misses)

        # all
        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        # to compare with ridge 9 one hot
        # encoded_x = pace.featurization.do_5d_encoding(x)

        self.clf = sklearn.linear_model.RidgeClassifier(alpha=1.0).fit(encoded_x, y)
        # self.clf = sklearn.linear_model.LogisticRegressionCV(Cs=8,cv=4).fit(encoded_x, y)


    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        
        
        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)

        # encoded_x = pace.featurization.do_5d_encoding(x)

        return self.clf.predict(encoded_x)



scores = pace.evaluate(RidgeAlgorithmFMLN5D,
                       **pace.load_data_set(16, peptide_lengths=[8, 9, 10, 11]))
pprint.pprint(scores)
