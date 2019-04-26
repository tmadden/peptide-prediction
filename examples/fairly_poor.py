import pace
import random
import pprint

# This is the example from the README.


class FairlyPoorAlgorithm(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        pass

    def predict(self, samples):
        return [1 if s.allele[0] == s.peptide[0] else 0 for s in samples]


# Evaluate our algorithm using PACE.
scores = pace.evaluate(FairlyPoorAlgorithm)
pprint.pprint(scores)
