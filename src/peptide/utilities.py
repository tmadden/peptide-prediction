import sklearn.preprocessing


amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def split_array(array, total_splits, split_index):
    start = len(array) * split_index // total_splits
    end = len(array) * (split_index + 1) // total_splits
    return (array[start:end], array[:start] + array[end:])


def create_one_hot_encoder(length):
    """
    Create an sklearn OneHotEncoder for encoding peptides.

    :param length: the expected length of the peptides to be encoded

    :returns: an sklearn.preprocessing.OneHotEncoder
    """

    return sklearn.preprocessing.OneHotEncoder(categories=[list(amino_acids)] * length)
