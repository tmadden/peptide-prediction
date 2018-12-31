import sklearn.linear_model

import numpy

import peptide.utilities, peptide.parsing, peptide.evaluation


class RidgeAlgorithm:
    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        y = [1] * len(hits) + [0] * len(misses)

        encoder = peptide.utilities.create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def eval(self, samples):
        x = [list(s.peptide) for s in samples]

        encoder = peptide.utilities.create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.clf.predict(encoded_x)


import random


class PureGuessingAlgorithm:
    def train(self, hits, misses):
        pass

    def eval(self, samples):
        return [random.uniform(0, 1) for _ in samples]


alleles = peptide.parsing.read_alleles_file("data/alleles_16.txt")
hits = list(peptide.parsing.read_hits_file("data/hits_16_9.txt"))
decoys = peptide.parsing.assign_alleles(
    alleles, peptide.parsing.read_decoys_file("data/decoys_9_train.txt")
)

import random

random.shuffle(decoys)
decoys = decoys[: len(hits)]

print("{} hits".format(len(hits)))
print("{} decoys".format(len(decoys)))

ridge_score = peptide.evaluation.evaluate(RidgeAlgorithm, hits, decoys)
print("ridge: {:.2f}".format(ridge_score))

guessing_score = peptide.evaluation.evaluate(PureGuessingAlgorithm, hits, decoys)
print("guess: {:.2f}".format(guessing_score))

