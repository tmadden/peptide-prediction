from pace.definitions import Sample

import itertools
import io
import random

from pkg_resources import resource_stream


def as_text_stream(stream):
    if not isinstance(stream, io.TextIOBase):
        stream = io.TextIOWrapper(stream)
    return stream


def read_hits_file(stream):
    def parse_line(line):
        fields = line.split()
        return Sample(allele=fields[0], peptide=fields[2])

    # The first line is always a header, so skip it.
    return map(parse_line, as_text_stream(stream).readlines()[1:])


def read_decoys_file(stream):
    return [line.strip() for line in as_text_stream(stream)]


def assign_alleles(alleles, decoys):

    return [
        Sample(allele=a, peptide=d)
        for a, d in zip(itertools.cycle(alleles), decoys)
    ]


def read_alleles_file(stream):
    return [line.strip() for line in as_text_stream(stream)]


def load_data_set(allele_count,
                  peptide_lengths=[8, 9, 10, 11],
                  nonbinder_fraction=0.5):

    alleles = read_alleles_file(
        resource_stream("pace", "data/alleles_{}.txt".format(allele_count)))

    # Load the hits file for each length.
    binders = []
    for length in peptide_lengths:
        binders.extend(
            read_hits_file(
                resource_stream(
                    "pace", "data/hits_{}_{}.txt".format(allele_count,
                                                         length))))
    # Load the decoys file for each length.
    nonbinders = []
    for length in peptide_lengths:
        nonbinders.extend(
            read_decoys_file(
                resource_stream("pace", "data/decoys_{}.txt".format(length))))
    nonbinders = assign_alleles(alleles, nonbinders)

    # Trim the decoys so that they represent the correct fraction.
    expected_nonbinder_count = int(
        len(binders) * (nonbinder_fraction / (1 - nonbinder_fraction)))
    random.shuffle(nonbinders)
    nonbinders = nonbinders[:expected_nonbinder_count]

    return {"binders": binders, "nonbinders": nonbinders}
