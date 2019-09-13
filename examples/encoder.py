import pace.sklearn, pace.featurization
import numpy as np


def fmln_plus(fmln_m, fmln_n, encoding_style, x):
    x = pace.featurization.do_FMLN_encoding(x, m=fmln_m, n=fmln_n)
    if encoding_style == 'one_hot':
        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        #print('length of x in fmln_plus is ' + str(len(x)))
        #print(type(x))
        xn = np.asarray(x)
        #print(xn.shape)
        if len(xn.shape) == 1:
            raise Exception(
                'Peptides not all the same length after fmln, probably a problem with input data.'
            )
        encoder.fit(xn)
        return encoder.transform(xn).toarray()
    elif encoding_style == '5d':
        return np.array(pace.featurization.do_5d_encoding(x), dtype='float64')
    elif encoding_style == 'both':
        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        ronehot = encoder.transform(x).toarray()
        r5d = pace.featurization.do_5d_encoding(x)
        return np.concatenate((ronehot, r5d), axis=1)
    else:
        raise Exception('Unknown encoding style used: ' + encoding_style)


#featurize test
'''
peplist = [['A', 'G', 'T', 'T'], ['Y', 'Y', 'K', 'K']]

x = pace.featurization.do_FMLN_encoding(peplist, m=2, n=2)

r5d = pace.featurization.do_5d_encoding(x)
print(r5d)

encoder = pace.sklearn.create_one_hot_encoder(len(peplist[0]))
encoder.fit(x)
ronehot = encoder.transform(x).toarray()
print(ronehot)
'''
