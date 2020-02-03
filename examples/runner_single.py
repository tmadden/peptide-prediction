from mixMHCpred import mixMHC
import numpy as np
import logging
import pace
import pprint

logging.basicConfig(level=logging.INFO)

scores = pace.evaluate(
    mixMHC,
    selected_lengths=[10],
    selected_alleles=['B4002'],
    dataset=pace.data.load_dataset(95),
    nbr_train=10,
    nbr_test=1000,
    random_seed=1)

pprint.pprint(scores)
#lambda: pan_length_rsd(encoding_style='one_hot'),