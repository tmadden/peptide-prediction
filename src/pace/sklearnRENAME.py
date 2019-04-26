"""This module provides various utilities for working with scikit-learn and
   PACE together."""

from pace.definitions import amino_acids


def create_one_hot_encoder(length):
    """
    Create an sklearn OneHotEncoder for encoding peptides.

    :param length: the expected length of the peptides to be encoded

    :returns: an sklearn.preprocessing.OneHotEncoder
    """
    import sklearn.preprocessing

    return sklearn.preprocessing.OneHotEncoder(
        categories=[list(amino_acids)] * length)
