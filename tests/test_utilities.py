import pace.utilities

import random

import numpy


def test_one_hot_encoding():
    encoder = pace.utilities.create_one_hot_encoder(10)

    x = [random.choices(pace.utilities.amino_acids, k=10)]

    encoder.fit(x)
    encoded_x = encoder.transform(x).toarray()

    assert encoded_x.shape == (1, 200)
    assert dict(zip(*numpy.unique(encoded_x, return_counts=True))) == {0: 190, 1: 10}


def test_split_array():
    from pace.utilities import split_array

    array = [1, 0, 2, 0, 3, 4, 0, 5, 0, 6]
    assert split_array(array, 4, 0) == ([1, 0], [2, 0, 3, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 1) == ([2, 0, 3], [1, 0, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 2) == ([4, 0], [1, 0, 2, 0, 3, 5, 0, 6])
    assert split_array(array, 4, 3) == ([5, 0, 6], [1, 0, 2, 0, 3, 4, 0])
