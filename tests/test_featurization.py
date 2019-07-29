import pace.featurization


def test_FMLN_encoding():

    peptides = ['ACDEFGIKL', 'MNPQRSTV']
    assert pace.featurization.do_FMLN_encoding(peptides, m=3,
                                               n=2) == ['ACDKL', 'MNPTV']


def test_5d_encoding():

    peptides = ['AC', 'M']
    assert pace.featurization.do_5d_encoding(peptides) == [[
        0.354311, 3.76204, -11.0357, -0.648649, 2.82792, -5.84613, 4.88503,
        1.62632, 9.39709, -5.84334
    ], [-10.585, -3.95856, -3.60113, 5.33888, 1.20304]]
