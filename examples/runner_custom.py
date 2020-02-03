from pan_length_stackdown import pan_length_rsd
from customA0203length9 import DJRSDfmln
import numpy as np
import logging
import pace
import pprint
import multiprocessing

logging.basicConfig(level=logging.INFO)

#shortcuts:
a0203 = ['A0203']
#there are three others close to A0203: 0201, 0204, and [farthest] 0207
a0203train = ['A0203']
a68 = ['A6802']
b35 = ['B3501']
a31 = ['A3101']


#wrapper function for pace.evaluate call:
def worker(rseed, return_dict):
    scores = pace.evaluate(
        lambda: DJRSDfmln(5, 4, encoding_style='one_hot'),
        #lambda: pan_length_rsd(encoding_style='one_hot'),
        selected_lengths=[9, 10, 11],
        selected_alleles=a31,
        test_alleles=a31,
        test_lengths=[10],
        dataset=pace.data.load_dataset(16),
        nbr_train=10,
        nbr_test=1000,
        random_seed=rseed)
    return_dict[rseed] = scores


#choose the set of random seeds
rseeds = range(10)

manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []

#run jobs in parallel
for rs in rseeds:
    p = multiprocessing.Process(target=worker, args=(rs, return_dict))
    jobs.append(p)
    p.start()

#wait for all runs to finish:
for proc in jobs:
    proc.join()

ppv_values = []
for r in return_dict.keys():
    s = return_dict[r]
    ppv_values.append(s['overall']['ppv'])
    print(s['overall']['ppv'])

mean_ppv = np.mean(ppv_values)
std_ppv = np.std(ppv_values)

print("Mean ppv is {:.2f}".format(mean_ppv))
print('Stdev of ppv is {:.3f}'.format(std_ppv))
