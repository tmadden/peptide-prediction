import pace, pace.sklearn, pace.featurization
import sklearn.linear_model
import pprint
import sklearn


class RidgeAlgorithm(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)

        x = pace.featurization.do_FMLN_encoding(x,m=6,n=5)

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        x = pace.featurization.do_FMLN_encoding(x,m=6,n=5)

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.clf.predict(encoded_x)


#scores = pace.evaluate(RidgeAlgorithm, selected_lengths=[8,9,10,11])
scores = pace.evaluate(RidgeAlgorithm, selected_lengths=[8,9,10,11],dataset=pace.data.load_dataset(95))
pprint.pprint(scores)
