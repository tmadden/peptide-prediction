import numpy as np
import pace.allele_similarity


def test_get_allele_similarity_mat():
    allele_similarity_name = 'motifs'
    allele_similarity = pace.allele_similarity.get_allele_similarity_mat(allele_similarity_name)
    assert allele_similarity.shape == (95, 95)


def test_get_similar_alleles_motifs():
    allele_similarity_name = 'motifs'
    allele = 'A2407'
    similarity_threshold = 0.8
    similar_alleles_thr = pace.allele_similarity.get_similar_alleles(
        allele_similarity_name, allele, similarity_threshold)
    assert np.all(similar_alleles_thr.index.values == ['A2407', 'A2301', 'A2402'])
    assert np.all(np.round(similar_alleles_thr[allele].values,4) == [1.0000, 0.9256, 0.8983])


def test_get_similar_alleles_pockets():
    allele_similarity_name = 'pockets'
    allele = 'A0101'
    similarity_threshold = 15
    similar_alleles_thr = pace.allele_similarity.get_similar_alleles(
        allele_similarity_name, allele, similarity_threshold)
    assert np.all(similar_alleles_thr.index.values == ['A0101', 'A3601'])
    assert np.all(np.round(similar_alleles_thr[allele].values,4) == [0.0000, 13.1073])


