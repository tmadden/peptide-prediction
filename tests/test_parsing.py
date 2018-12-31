import peptide.parsing


def test_decoy_parsing():
    decoys = peptide.parsing.read_decoys_file("data/decoys_9_train.txt")
    assert len(decoys) == 999995
    assert decoys[:4] == ["RISLRKVRS", "LNGSKLWIS", "LTMAVLHVT", "LPEGSKDSF"]


def test_hits_parsing():
    hits = list(peptide.parsing.read_hits_file("data/hits_16_9.txt"))
    assert len(hits) == 16062
    assert hits[:4] == [
        peptide.parsing.Sample(allele="A0101", peptide="AADIFYSRY"),
        peptide.parsing.Sample(allele="A0101", peptide="AADLNLVLY"),
        peptide.parsing.Sample(allele="A0101", peptide="AADLVEALY"),
        peptide.parsing.Sample(allele="A0101", peptide="AIDEDVLRY"),
    ]


def test_alleles_parsing():
    alleles = list(peptide.parsing.read_alleles_file("data/alleles_16.txt"))
    assert len(alleles) == 16
    assert alleles[:4] == ["A0101", "A0201", "A0203", "A0204"]


def test_allele_assignment():
    alleles = ["A", "B", "C"]
    peptides = ["T", "U", "V", "W", "X", "Y", "Z"]
    assert peptide.parsing.assign_alleles(alleles, peptides) == [
        peptide.parsing.Sample(allele="A", peptide="T"),
        peptide.parsing.Sample(allele="B", peptide="U"),
        peptide.parsing.Sample(allele="C", peptide="V"),
        peptide.parsing.Sample(allele="A", peptide="W"),
        peptide.parsing.Sample(allele="B", peptide="X"),
        peptide.parsing.Sample(allele="C", peptide="Y"),
        peptide.parsing.Sample(allele="A", peptide="Z"),
    ]
