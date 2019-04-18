from pace.definitions import Sample

import itertools
import io
import random

from pkg_resources import resource_stream
from pace.definitions import DataSet


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
    stream = as_text_stream(stream)
    while True:
        for line in stream:
            yield line.strip()
        # We ran out of decoys, so reset to the beginning of the stream.
        stream.seek(0, 0)


def read_alleles_file(stream):
    return (line.strip() for line in as_text_stream(stream))


class BuiltinDataSet(DataSet):
    def __init__(self, allele_count):
        self.allele_count = allele_count

    def get_binders(self, length):
        return read_hits_file(
            resource_stream(
                "pace", "data/hits_{}_{}.txt".format(self.allele_count,
                                                     length)))

    def get_nonbinders(self, length):
        return read_decoys_file(
            resource_stream("pace", "data/decoys_{}.txt".format(length)))


def load_dataset(allele_count):
    return BuiltinDataSet(allele_count)
