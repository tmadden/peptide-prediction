from pace.definitions import Sample

import itertools
import io


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
