from pan_length_stackdown import pan_length_rsd
from customA0203length9 import DJRSDfmln
import numpy as np
import logging
import pace
import pprint

logging.basicConfig(level=logging.INFO)

#shortcuts:
a0203 = ['A0203']
#there are three others close to A0203: 0201, 0204, and [farthest] 0207
a0203train = ['A0203']
a68 = ['A6802']
b35 = ['B3501']
a31 = ['A3101']

for i in range(10):
    scores = pace.evaluate(
        lambda: DJRSDfmln(4, 4, encoding_style='one_hot'),
        selected_lengths=[9, 10, 11],
        selected_alleles=a31,
        test_alleles=a31,
        test_lengths=[10],
        dataset=pace.data.load_dataset(16),
        nbr_train=10,
        nbr_test=1000,
        random_seed=i + 6)

    pprint.pprint(scores)

#lambda: pan_length_rsd(encoding_style='one_hot'),