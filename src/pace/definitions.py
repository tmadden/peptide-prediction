from abc import ABC, abstractmethod
from typing import NamedTuple

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

peptide_lengths = [8, 9, 10, 11]


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

        :returns: an array-like value containing a prediction for each sample -
        Each prediction is a number between 0 and 1 indicating how likely the
        sample is to bind.
        """
        pass  # pragma: no cover


class DataSet(ABC):
    @abstractmethod
    def get_binders(self, length):
        """
        Get all binders with the specified length.

        Note that this is allowed to return a single-use iterable.

        :param length: the peptide length the caller is interested in

        :returns: all binders with that length
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_nonbinders(self, length):
        """
        Get all nonbinders with the specified length.

        Note that this is allowed to return a single-use iterable.

        :param length: the peptide length the caller is interested in

        :returns: all nonbinders with that length
        """
        pass  # pragma: no cover
