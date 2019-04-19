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

        Parameters
        ----------
        binders
            samples that are known to bind
        nonbinders
            samples that are known to not bind
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, samples):
        """
        Predict whether or not a list of samples will bind.

        Parameters
        ----------
        samples
            the samples to predict

        Returns
        -------
        an array-like value containing a prediction for each sample - Each
        prediction is a number between 0 and 1 indicating how likely the sample
        is to bind.
        """
        pass  # pragma: no cover


class DataSet(ABC):
    @abstractmethod
    def get_binders(self, length):
        """
        Get all binders with the specified length.

        Parameters
        ----------
        length
            the peptide length the caller is interested in

        Returns
        -------
        all binders with that length - Note that this is allowed to return a
        single-use iterable.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_nonbinders(self, length):
        """
        Get all nonbinders with the specified length.

        Parameters
        ----------
        length
            the peptide length the caller is interested in

        Returns
        -------
        all binders with that length - Note that this is allowed to return a
        single-use iterable.
        """
        pass  # pragma: no cover
