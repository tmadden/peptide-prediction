import pace.featurization

def test_FMLN_encoding():
    
    peptides = ['ACDEFGIKL', 'MNPQRSTV']
    assert pace.featurization.do_FMLN_encoding(peptides,m=3,n=2) == [
        'ACDKL',
        'MNPTV'
    ]

def test_5d_encoding():
    
    peptides = ['AC', 'M']
    assert pace.featurization.do_5d_encoding(peptides) == [
        [0.354311, 3.76204, -11.0357, -0.648649, 2.82792, 
        -5.84613, 4.88503, 1.62632, 9.39709, -5.84334],
        [-10.585, -3.95856, -3.60113, 5.33888, 1.20304]
    ]


def test_vedAll():
    
    assert pace.featurization.get_vedAll(16) == [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 
        1, 1, 1, 2, 1, 2, 2, 4, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 3, 1, 3, 3, 5, 1, 2, 4, 3, 1, 2, 3, 1, 3, 
        3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 4, 1, 5, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 5, 1, 
        4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 3, 1, 1, 
        1, 5, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

def test_snpSets():
    
    assert pace.featurization.get_snpSets(95)[0] == [32, 45, 46, 52, 62, 63, 103, 107, 163, 166, 167, 171, 177, 178, 180]
        
