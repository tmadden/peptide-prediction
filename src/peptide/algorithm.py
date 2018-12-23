from abc import ABC, abstractmethod

from collections import namedtuple

TrainingData = namedtuple("TrainingData", ["", ""])

Sample = namedtuple("Sample", ["allele", "peptide"])


class PredictionAlgorithm(ABC):
    @abstractmethod
    def train(self, hits, misses):
        """
        Train this instance using the supplied training data.

        """
        pass

    @abstractmethod
    def predict(self, peptide):
        pass

    def predict_list(self, peptides):
        return [self.predict(p) for p in peptides]

