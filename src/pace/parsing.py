import pace.utilities
from pace.definitions import Sample

import itertools


def read_hits_file(path):
    with open(path, "r") as file:

        def parse_line(line):
            fields = line.split()
            return Sample(allele=fields[0], peptide=fields[2])

        # The first line is always a header, so skip it.
        return map(parse_line, file.readlines()[1:])


def read_decoys_file(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def assign_alleles(alleles, decoys):

    return [
        Sample(allele=a, peptide=d)
        for a, d in zip(itertools.cycle(alleles), decoys)
    ]


def read_alleles_file(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]
