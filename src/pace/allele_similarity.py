import numpy as np
import pandas as pd
import pace
from pace.definitions import amino_acids, builtin_aa_encodings, builtin_allele_similarities
from pace.sklearn import create_one_hot_encoder
from pkg_resources import resource_stream


def load_allele_similarity(allele_similarity_name):
    allele_similarity = pd.read_csv(
        resource_stream("pace", "data/allele_similarity_mat_{}.txt".format(allele_similarity_name)),
        sep=' ', index_col=0)
    return allele_similarity


def get_allele_similarity_mat(allele_similarity_name):
    """
    Get a matrix of pre-computed allele similarities

    Parameters
    ----------
    allele_similarity_name : str
        Pre-computed allele similarity matrices are availble based on 
        observed peptide binding motifs ('motifs') or HLA protein binding 
        pocket residues ('pockets').

    Returns
    -------
    pandas.core.frame.DataFrame
        allele similarity matrix
    """
    return load_allele_similarity(allele_similarity_name)


def get_similar_alleles(allele_similarity_name, allele, similarity_threshold):
    """
    Get the most similar alleles to a given allele, based on a specified 
    allele similarity matrix and similarity threshold.

    Parameters
    ----------
    allele_similarity_name : str
        Pre-computed allele similarity matrices are availble based on 
        observed peptide binding motifs ('motifs') or HLA protein binding 
        pocket residues ('pockets').

    allele : str
        The allele for which to determine similar alleles

    similarity_threshold
        Numerical threhosld value that determins the cutoff for considering 
        an allele similar to the given allele.

    Returns
    -------
    pandas.core.frame.DataFrame
        The similar alleles satisfying the specifid threshold along 
        with the numerical similarity values. Note that the given allele 
        is also returned.
    """
    assert(allele_similarity_name in builtin_allele_similarities)
    allele_similarity = get_allele_similarity_mat(allele_similarity_name)

    similar_alleles = allele_similarity[allele]
    if allele_similarity_name == 'motifs': # higher values => more similar alleles
        similar_alleles_thr = similar_alleles[similar_alleles > similarity_threshold]
        similar_alleles_thr = similar_alleles_thr[(similar_alleles_thr*-1).argsort()]
    if allele_similarity_name == 'pockets': # higher values => less similar alleles
        similar_alleles_thr = similar_alleles[similar_alleles < similarity_threshold]
        similar_alleles_thr = similar_alleles_thr[similar_alleles_thr.argsort()]
    return similar_alleles_thr.to_frame()




