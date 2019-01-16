import pace.data
from pace import Sample

from pkg_resources import resource_stream


def test_decoy_parsing():
    decoys = pace.data.read_decoys_file(
        resource_stream("pace", "data/decoys_9.txt"))
    assert len(decoys) == 982791
    assert decoys[:4] == ["RISLRKVRS", "LNGSKLWIS", "LTMAVLHVT", "LPEGSKDSF"]


def test_hits_parsing():
    hits = list(
        pace.data.read_hits_file(
            resource_stream("pace", "data/hits_16_9.txt")))
    assert len(hits) == 16062
    assert hits[:4] == [
        Sample(allele="A0101", peptide="AADIFYSRY"),
        Sample(allele="A0101", peptide="AADLNLVLY"),
        Sample(allele="A0101", peptide="AADLVEALY"),
        Sample(allele="A0101", peptide="AIDEDVLRY"),
    ]


def test_alleles_parsing():
    alleles = list(
        pace.data.read_alleles_file(
            resource_stream("pace", "data/alleles_16.txt")))
    assert len(alleles) == 16
    assert alleles[:4] == ["A0101", "A0201", "A0203", "A0204"]


def test_allele_assignment():
    alleles = ["A", "B", "C"]
    peptides = ["T", "U", "V", "W", "X", "Y", "Z"]
    assert pace.data.assign_alleles(alleles, peptides) == [
        Sample(allele="A", peptide="T"),
        Sample(allele="B", peptide="U"),
        Sample(allele="C", peptide="V"),
        Sample(allele="A", peptide="W"),
        Sample(allele="B", peptide="X"),
        Sample(allele="C", peptide="Y"),
        Sample(allele="A", peptide="Z"),
    ]


def test_data_set():
    data_set = pace.data.load_data_set(
        16, peptide_lengths=[9, 10], nonbinder_fraction=0.75)

    binders = data_set["binders"]
    nonbinders = data_set["nonbinders"]
    all_samples = binders + nonbinders

    assert len(nonbinders) == len(binders) * 3

    assert all(len(s.peptide) in (9, 10) for s in all_samples)

    assert len(set(s.allele for s in all_samples)) == 16
