import pace.utilities


def test_split_array():
    from pace.utilities import split_array

    array = [1, 0, 2, 0, 3, 4, 0, 5, 0, 6]
    assert split_array(array, 4, 0) == ([1, 0], [2, 0, 3, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 1) == ([2, 0, 3], [1, 0, 4, 0, 5, 0, 6])
    assert split_array(array, 4, 2) == ([4, 0], [1, 0, 2, 0, 3, 5, 0, 6])
    assert split_array(array, 4, 3) == ([5, 0, 6], [1, 0, 2, 0, 3, 4, 0])
