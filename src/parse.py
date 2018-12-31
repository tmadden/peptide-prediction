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


alleles = peptide.parsing.read_alleles_file("../data/alleles_16.txt")

hits = list(peptide.parsing.read_hits_file("../data/hits_16_9.txt"))
decoys = peptide.parsing.assign_alleles(
    alleles, peptide.parsing.read_decoys_file("../data/decoys_9_train.txt")
)

import random

random.shuffle(decoys)
decoys = decoys[:40000]

print(len(hits))
print(len(decoys))

peptide.evaluation.evaluate(RidgeAlgorithm, hits, decoys)

# hit_split = len(hits) // 2
# decoy_split = len(decoys) // 2

# training_hits = hits[:hit_split]
# training_decoys = decoys[:decoy_split]

# algorithm = RidgeAlgorithm()
# algorithm.train(training_hits, training_decoys)

# eval_hits = hits[hit_split:]
# eval_decoys = decoys[decoy_split:]

# print(peptide.evaluation.score(algorithm, training_hits, training_decoys))
# print(peptide.evaluation.score(algorithm, eval_hits, eval_decoys))
