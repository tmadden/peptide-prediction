from abc import ABC, abstractmethod
from typing import NamedTuple


class Sample(NamedTuple):
    allele: str
    peptide: str


class PredictionAlgorithm(ABC):
    @abstractmethod
    def train(self, binders, nonbinders):
        """
        Train this instance using the supplied training data.

        :param binders: samples that are known to bind
        :param nonbinders: samples that are known to not bind
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, samples):
        """
        Predict whether or not a list of samples will bind.

        :param samples: samples to predict

        :returns: an array-like value containing a prediction for each sample - Each prediction is a number between 0 and 1 indicating how likely the sample is to bind.
        """
        pass  # pragma: no cover
