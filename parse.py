import random

import sklearn.preprocessing

import sklearn.linear_model

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

amino_acid_index_lookup = {c: i for i, c in enumerate(amino_acids)}

from collections import namedtuple

Sample = namedtuple("Sample", ["allele", "peptide"])


def create_one_hot_encoder(length):
    """
    Create an sklearn OneHotEncoder for encoding peptides.

    :param length: the expected length of the peptides to be encoded

    :returns: an sklearn.preprocessing.OneHotEncoder
    """

    return sklearn.preprocessing.OneHotEncoder(categories=[list(amino_acids)] * length)


def read_hits_file(path):
    with open(path, "r") as file:

        def read_line(line):
            fields = line.split()
            return Sample(allele=fields[0], peptide=fields[2])

        # The first line is always a header, so skip it.
        return map(read_line, file.readlines()[1:])


def read_decoys_file(path):
    with open(path, "r") as file:
        # The last line is always a header, so skip it.
        return [line.strip() for line in file.readlines()[:-1]]

def assign_random_alleles(alleles, decoys):
    return [Sample(allele=random.choice(alleles), peptide=d) for d in decoys]

def read_alleles_file(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def tag_decoys(alleles, decoys):
    return [Sample(allele=a, peptide=d) for a in alleles for d in decoys]


class RidgeAlgorithm:
    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        y = [1] * len(hits) + [0] * len(misses)

        encoder = create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def eval(self, samples):
        x = [list(s.peptide) for s in samples]

        encoder = create_one_hot_encoder(9)
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.clf.predict(encoded_x)


import random


def score(algorithm, hits, misses):
    paired_samples = list(zip(hits + misses, [1] * len(hits) + [0] * len(misses)))
    random.shuffle(paired_samples)
    x = [sample for sample, score in paired_samples]
    correct_scores = [score for sample, score in paired_samples]
    predicted_scores = algorithm.eval(x)
    score = sum(1 for (p, s) in zip(correct_scores, predicted_scores) if p == s) / (
        len(hits) + len(misses)
    )
    return score

alleles = read_alleles_file("data/alleles_16.txt")

hits = list(read_hits_file("data/hits_16_9.txt"))
decoys = assign_random_alleles(alleles, [s.replace("X", "A") for s in read_decoys_file("data/decoys_9_train.txt")])

random.shuffle(decoys)
decoys = decoys[: len(hits)]

# from functools import reduce

# unique_hits = reduce(lambda u, h: u.union(set(h)), hits, set())
# print(len(unique_hits))
# print(sorted(list(unique_hits)))

# unique_decoys = reduce(lambda u, d: u.union(set(d)), decoys, set())
# print(len(unique_decoys))
# print(sorted(list(unique_decoys)))


hit_split = len(hits) // 2
decoy_split = len(decoys) // 2

training_hits = hits[:hit_split]
training_decoys = decoys[:decoy_split]

# tx = training_hits + training_decoys  # tag_decoys(alleles, decoys)
# ty = [1] * len(training_hits) + [0] * len(training_decoys)
# encoder.fit(tx)
# encoded_tx = encoder.transform(tx).toarray()

algorithm = RidgeAlgorithm()
algorithm.train(training_hits, training_decoys)

eval_hits = hits[hit_split:]
eval_decoys = decoys[decoy_split:]

print(score(algorithm, training_hits, training_decoys))
print(score(algorithm, eval_hits, eval_decoys))

# ex = eval_hits + eval_decoys  # tag_decoys(alleles, decoys)
# ey = [1] * len(eval_hits) + [0] * len(eval_decoys)
# encoder.fit(ex)
# encoded_ex = encoder.transform(ex).toarray()

# print(encoded_tx.shape)
# print(len(encoded_tx[0]))
# print(len([i for i in encoded_tx[0] if i != 0]))

# print(encoded_ex.shape)
# print(len(encoded_ex[0]))
# print(len([i for i in encoded_ex[0] if i != 0]))

# clf = sklearn.linear_model.RidgeClassifier().fit(encoded_tx, ty)
# print(clf.score(encoded_tx, ty))
# print(clf.score(encoded_ex, ey))
