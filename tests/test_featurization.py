import numpy as np
import pandas as pd
import pace.featurization

def test_load_aafeatmat():
    aafeatmat_name = 'BLOSUM62'
    aafeatmat = pace.featurization.load_aafeatmat(aafeatmat_name)
    assert aafeatmat.shape[0] == 20
    assert all([aa in list(pace.definitions.amino_acids) for aa in aafeatmat.index.values])


def test_encode_onehot():
    from pace.featurization import encode
    sequences = [
            "AADIFYSRY",
            "AADLNLVLY",
            "AAAAAAACL",
            "WIDEDVLRY"]
    
    encoding = encode(sequences)
    assert encoding.shape[0] == len(sequences)
    assert encoding.shape[1] % len(sequences[0]) == 0
    assert np.all(encoding[0,:] == [
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1.
    ])
    assert np.all(encoding[-1,:] == [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1.
    ])


def test_encode():
    from pace.featurization import encode
    sequences = [
            "AADIFYSRY",
            "AADLNLVLY",
            "AAAAAAACL",
            "WIDEDVLRY"]
    
    encoding = encode(sequences, 'BLOSUM62')
    assert encoding.shape[0] == len(sequences)
    assert encoding.shape[1] % len(sequences[0]) == 0
    assert np.all(encoding[0,:] == [
          4., -1., -2., -2.,  0., -1., -1.,  0., -2., -1., -1., -1., -1.,
         -2., -1.,  1.,  0., -3., -2.,  0.,  4., -1., -2., -2.,  0., -1.,
         -1.,  0., -2., -1., -1., -1., -1., -2., -1.,  1.,  0., -3., -2.,
          0., -2., -2.,  1.,  6., -3.,  0.,  2., -1., -1., -3., -4., -1.,
         -3., -3., -1.,  0., -1., -4., -3., -3., -1., -3., -3., -3., -1.,
         -3., -3., -4., -3.,  4.,  2., -3.,  1.,  0., -3., -2., -1., -3.,
         -1.,  3., -2., -3., -3., -3., -2., -3., -3., -3., -1.,  0.,  0.,
         -3.,  0.,  6., -4., -2., -2.,  1.,  3., -1., -2., -2., -2., -3.,
         -2., -1., -2., -3.,  2., -1., -1., -2., -1.,  3., -3., -2., -2.,
          2.,  7., -1.,  1., -1.,  1.,  0., -1.,  0.,  0.,  0., -1., -2.,
         -2.,  0., -1., -2., -1.,  4.,  1., -3., -2., -2., -1.,  5.,  0.,
         -2., -3.,  1.,  0., -2.,  0., -3., -2.,  2., -1., -3., -2., -1.,
         -1., -3., -2., -3., -2., -2., -2., -3., -2., -1., -2., -3.,  2.,
         -1., -1., -2., -1.,  3., -3., -2., -2.,  2.,  7., -1.
    ])
    assert np.all(encoding[-1,:] == [
         -3., -3., -4., -4., -2., -2., -3., -2., -2., -3., -2., -3., -1.,
          1., -4., -3., -2., 11.,  2., -3., -1., -3., -3., -3., -1., -3.,
         -3., -4., -3.,  4.,  2., -3.,  1.,  0., -3., -2., -1., -3., -1.,
          3., -2., -2.,  1.,  6., -3.,  0.,  2., -1., -1., -3., -4., -1.,
         -3., -3., -1.,  0., -1., -4., -3., -3., -1.,  0.,  0.,  2., -4.,
          2.,  5., -2.,  0., -3., -3.,  1., -2., -3., -1.,  0., -1., -3.,
         -2., -2., -2., -2.,  1.,  6., -3.,  0.,  2., -1., -1., -3., -4.,
         -1., -3., -3., -1.,  0., -1., -4., -3., -3.,  0., -3., -3., -3.,
         -1., -2., -2., -3., -3.,  3.,  1., -2.,  1., -1., -2., -2.,  0.,
         -3., -1.,  4., -1., -2., -3., -4., -1., -2., -3., -4., -3.,  2.,
          4., -2.,  2.,  0., -3., -2., -1., -2., -1.,  1., -1.,  5.,  0.,
         -2., -3.,  1.,  0., -2.,  0., -3., -2.,  2., -1., -3., -2., -1.,
         -1., -3., -2., -3., -2., -2., -2., -3., -2., -1., -2., -3.,  2.,
         -1., -1., -2., -1.,  3., -3., -2., -2.,  2.,  7., -1.
    ])