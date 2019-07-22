from abc import ABC, abstractmethod
from typing import NamedTuple

# All public definitions from this module are exposed as part of the package-
# level API for PACE, so it's important to keep the list clean.
__all__ = [
    'amino_acids', 'peptide_lengths', 'Sample', 'Dataset',
    'PredictionAlgorithm', 'PredictionResult', 'Scorer'
]

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

peptide_lengths = [8, 9, 10, 11]


class Sample(NamedTuple):
    """
    a sample to predict
    """

    allele: str
    """
    the allele code for the MHC molecule
    """

    peptide: str
    """
    the amino acid sequence for the peptide (as a string)
    """


class Dataset(ABC):
    """
    an abstract base class defining the interface required of a dataset
    """

    @abstractmethod
    def get_binders(self, length):
        """
        Get all binders with the specified length.

        Parameters
        ----------
        length : int
            the peptide length the caller is interested in

        Returns
        -------
        Iterable[pace.Sample]
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
        length : int
            the peptide length the caller is interested in

        Returns
        -------
        List[str]
            all non-binder peptides with that length.
        """
        pass  # pragma: no cover


class PredictionAlgorithm(ABC):
    """
    an abstract base class defining the interface required of prediction
    algorithms that are to be evaluated by PACE
    """

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
        samples : List[pace.Sample]
            the samples to predict

        Returns
        -------
        NumPy array-like object (e.g., list of floats)
            predictions for each sample - Each prediction is a number between
            0 and 1 indicating how likely the sample is to bind.
        """
        pass  # pragma: no cover


class PredictionResult(NamedTuple):
    """
    the result of predicting a single sample
    """

    sample: Sample
    """
    the sample that was predicted
    """

    prediction: float
    """
    the algorithm's prediction (between 0 and 1)
    """

    truth: float
    """
    the true answer (either 0 or 1)
    """


class Scorer(ABC):
    """
    an abstract base class defining the interface required of scorers - A scorer
    quantifies (or summarizes) the accuracy of prediction results.
    """

    @abstractmethod
    def score(self, results):
        """
        Generate the score for a set of prediction results.

        Parameters
        ----------
        results : Iterable[PredictionResult]
            the prediction results to score

        Returns
        -------
        Any
            whatever summary info the scorer would like to generate for the
            results
        """
        pass  # pragma: no cover
