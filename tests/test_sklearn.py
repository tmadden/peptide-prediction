import pace
from pace.sklearn import create_one_hot_encoder

import random
import numpy


def test_one_hot_encoding():
    encoder = create_one_hot_encoder(10)

    x = [random.choices(pace.amino_acids, k=10)]

    encoder.fit(x)
    encoded_x = encoder.transform(x).toarray()

    assert encoded_x.shape == (1, 200)
    assert dict(zip(*numpy.unique(encoded_x, return_counts=True))) == {
        0: 190,
        1: 10
    }
