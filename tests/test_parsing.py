import pace.parsing
from pace.parsing import Sample


def test_decoy_parsing():
    decoys = pace.parsing.read_decoys_file("data/decoys_9_train.txt")
    assert len(decoys) == 999995
    assert decoys[:4] == ["RISLRKVRS", "LNGSKLWIS", "LTMAVLHVT", "LPEGSKDSF"]


def test_hits_parsing():
    hits = list(pace.parsing.read_hits_file("data/hits_16_9.txt"))
    assert len(hits) == 16062
    assert hits[:4] == [
        Sample(allele="A0101", peptide="AADIFYSRY"),
        Sample(allele="A0101", peptide="AADLNLVLY"),
        Sample(allele="A0101", peptide="AADLVEALY"),
        Sample(allele="A0101", peptide="AIDEDVLRY"),
    ]


def test_alleles_parsing():
    alleles = list(pace.parsing.read_alleles_file("data/alleles_16.txt"))
    assert len(alleles) == 16
    assert alleles[:4] == ["A0101", "A0201", "A0203", "A0204"]


def test_allele_assignment():
    alleles = ["A", "B", "C"]
    peptides = ["T", "U", "V", "W", "X", "Y", "Z"]
    assert pace.parsing.assign_alleles(alleles, peptides) == [
        Sample(allele="A", peptide="T"),
        Sample(allele="B", peptide="U"),
        Sample(allele="C", peptide="V"),
        Sample(allele="A", peptide="W"),
        Sample(allele="B", peptide="X"),
        Sample(allele="C", peptide="Y"),
        Sample(allele="A", peptide="Z"),
    ]
