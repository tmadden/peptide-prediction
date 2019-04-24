import pace.data
from pace import Sample

from pkg_resources import resource_stream


def test_decoy_parsing():
    decoys = pace.data.read_decoys_file(
        resource_stream("pace", "data/decoys_9.txt"))
    assert next(decoys) == "RISLRKVRS"
    assert next(decoys) == "LNGSKLWIS"
    assert next(decoys) == "LTMAVLHVT"
    assert next(decoys) == "LPEGSKDSF"
    # Test that decoys wrap.
    for i in range(0, 982787):
        next(decoys)
    assert next(decoys) == "RISLRKVRS"
    assert next(decoys) == "LNGSKLWIS"


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
