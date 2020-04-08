from isolated_singles import isolated_alg
import numpy as np
import logging
import pace
import pprint
import multiprocessing
from pkg_resources import resource_stream

logging.basicConfig(level=logging.INFO)


#wrapper function for pace.evaluate call:
def worker(test_allele, train_alleles, test_length, train_lengths, fmln_m,
           fmln_n, rseed, return_dict):
    scoresALG = pace.evaluate(
        lambda: isolated_alg(fmln_m, fmln_n, encoding_style='one_hot'),
        selected_lengths=train_lengths,
        selected_alleles=train_alleles,
        test_alleles=test_allele,
        test_lengths=test_length,
        dataset=pace.data.load_dataset(16),
        nbr_train=10,
        nbr_test=1000,
        random_seed=rseed)

    return_dict[rseed] = scoresALG


#choose the set of random seeds
rseeds = range(10)

manager = multiprocessing.Manager()

import pace.data
alleles = list(
    pace.data.read_alleles_file(
        #note if you change 16 to 95 gotta change it above too.
        resource_stream("pace", "data/alleles_16.txt")))

lengths = [8, 9, 10, 11]

meanppvALG = np.zeros(shape=(len(alleles), len(lengths)))
stdppvALG = np.zeros(shape=(len(alleles), len(lengths)))

for ia in range(len(alleles)):
    for il in range(len(lengths)):
        m = 4
        n = lengths[il] - m

        return_dict = manager.dict()
        jobs = []

        #run jobs in parallel
        for rs in rseeds:
            p = multiprocessing.Process(
                target=worker,
                args=([alleles[ia]], [alleles[ia]], [lengths[il]],
                      [lengths[il]], m, n, rs, return_dict))
            jobs.append(p)
            p.start()

        #wait for all runs to finish:
        for proc in jobs:
            proc.join()

        ppv_valuesALG = []

        for r in return_dict.keys():
            s = return_dict[r]
            ppv_valuesALG.append(s['overall']['ppv'])

        print("allele " + alleles[ia] + ", length " + str(lengths[il]))

        mean_ppvALG = np.mean(ppv_valuesALG)
        std_ppvALG = np.std(ppv_valuesALG)
        print("  Mean ppv is {:.2f}".format(mean_ppvALG))
        print("  Stdev of ppv is {:.3f}".format(std_ppvALG))
        meanppvALG[ia, il] = mean_ppvALG
        stdppvALG[ia, il] = std_ppvALG

np.savetxt('mean_ppv_alg.csv', meanppvALG)
np.savetxt('std_ppv_alg.csv', stdppvALG)
