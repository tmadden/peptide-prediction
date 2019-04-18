import pace
import random
import pprint


class PureGuessingAlgorithm(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        pass

    def predict(self, samples):
        return [random.uniform(0, 1) for _ in samples]


scores = pace.evaluate(PureGuessingAlgorithm,
                       **pace.load_dataset(16, nonbinder_fraction=0.9))
pprint.pprint(scores)
