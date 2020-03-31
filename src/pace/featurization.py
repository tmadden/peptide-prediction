import numpy as np
import pandas as pd
import pace
from pace.definitions import amino_acids, builtin_aa_encodings, builtin_allele_similarities
from pace.sklearn import create_one_hot_encoder
from pkg_resources import resource_stream


def load_aafeatmat(aafeatmat_name):
    aafeatmat = pd.read_csv(
        resource_stream("pace", "data/aafeatmat_{}.txt".format(aafeatmat_name)),
        sep='\t', index_col=0)
    return aafeatmat


def encode_onehot(sequences, pep_len):
    encoder = pace.sklearn.create_one_hot_encoder(pep_len)
    encoder.fit(sequences)
    encoded = encoder.transform(sequences).toarray()
    return encoded


def encode(sequences, aafeatmat="onehot"):
    """
    Create a numerical encoding for the input peptide sequences
    Assumes that all input sequences have the same length 
    (TO DO: how should we integrate error handling?)

    Parameters
    ----------
    sequences
        List of peptide sequences. A list of strings is accepted as well 
        as a list of lists where the inner lists are single amino acids. 
        All sequences need to be the same length.

    aafeatmat
        Either the name of one of the builtin peptide encodings or a pandas 
        DataFrame with one amino acid per row, and columns with features. 
        (Rows: 20 amino acids; columns: the encoding of each amino acid.)

    Returns
    -------
    numpy.ndarray
        encoded sequences
    """

    # Split up each peptide string into individual amino acids
    if isinstance(sequences[0], str):
        sequences = [list(s) for s in sequences]

    # Input peptide sequences need to be of the same length
    lens = [len(s) for s in sequences]
    pep_len = lens[0]
    assert(all(l==pep_len for l in lens)) # how should we integrate error handling?

    # One-hot (binary) encoding
    encoded = encode_onehot(sequences, pep_len)

    if aafeatmat == None or aafeatmat == 'onehot' or aafeatmat == 'binary':
         return encoded

    # Transform one-hot encoding to specified encoding type
    if isinstance(aafeatmat, str):
        assert(aafeatmat in builtin_aa_encodings)
        aafeatmat = load_aafeatmat(aafeatmat)
    # Ensure the rows have the same order of amino acids as 
    # amino_acids in pace.definitions (and onehot encoding). 
    # This enables efficient transfprmation to other encodings 
    # by multiplication (below).
    aafeatmat = aafeatmat.loc[list(amino_acids),:]
    # Block diagnoal aafeatmat
    aafeatmat_bd = np.kron(np.eye(pep_len, dtype=int), aafeatmat)
    # Feature encoding (@ matrix multiplication)
    return encoded @ aafeatmat_bd 


