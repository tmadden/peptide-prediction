from DonJulioNoir import DJRSD, DJRSDfmln
from voting_fmln import voting_fmln
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
    scoresDJRSD = pace.evaluate(
        lambda: DJRSDfmln(fmln_m, fmln_n, encoding_style='one_hot'),
        selected_lengths=train_lengths,
        selected_alleles=train_alleles,
        test_alleles=test_allele,
        test_lengths=test_length,
        dataset=pace.data.load_dataset(95),
        nbr_train=10,
        nbr_test=1000,
        random_seed=rseed)
    scoresVOTING = pace.evaluate(
        lambda: voting_fmln(fmln_m, fmln_n, encoding_style='one_hot'),
        selected_lengths=train_lengths,
        selected_alleles=train_alleles,
        test_alleles=test_allele,
        test_lengths=test_length,
        dataset=pace.data.load_dataset(95),
        nbr_train=10,
        nbr_test=1000,
        random_seed=rseed)

    return_dict[rseed] = scoresDJRSD
    return_dict[rseed + 100] = scoresVOTING


#choose the set of random seeds
rseeds = range(10)

#do i need to put this inside the loops?
manager = multiprocessing.Manager()

import pace.data
alleles = list(
    pace.data.read_alleles_file(
        resource_stream("pace", "data/alleles_16.txt")))

lengths = [8, 9, 10, 11]

meanppvNN = np.zeros(shape=(len(alleles), len(lengths)))
stdppvNN = np.zeros(shape=(len(alleles), len(lengths)))
meanppvVOTING = np.zeros(shape=(len(alleles), len(lengths)))
stdppvVOTING = np.zeros(shape=(len(alleles), len(lengths)))

for ia in range(len(alleles)):
    for il in range(len(lengths)):
        m = 4
        n = lengths[il] - m

        return_dict = manager.dict()
        jobs = []

        #def worker(test_allele, train_alleles, test_length, train_lengths, fmln_m,
        #   fmln_n, rseed, return_dict):

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

        ppv_valuesNN = []
        ppv_valuesVOTING = []

        for r in return_dict.keys():
            s = return_dict[r]
            if r < 90:
                ppv_valuesNN.append(s['overall']['ppv'])
            else:
                ppv_valuesVOTING.append(s['overall']['ppv'])

        print("allele " + alleles[ia] + ", length " + str(lengths[il]))

        mean_ppvNN = np.mean(ppv_valuesNN)
        std_ppvNN = np.std(ppv_valuesNN)
        print("  Mean ppv NN is {:.2f}".format(mean_ppvNN))
        print("  Stdev of ppv NN is {:.3f}".format(std_ppvNN))
        meanppvNN[ia, il] = mean_ppvNN
        stdppvNN[ia, il] = std_ppvNN

        mean_ppvVOTING = np.mean(ppv_valuesVOTING)
        std_ppvVOTING = np.std(ppv_valuesVOTING)
        print("  Mean ppv VOTING is {:.2f}".format(mean_ppvVOTING))
        print("  Stdev of ppv VOTING is {:.3f}".format(std_ppvVOTING))
        meanppvVOTING[ia, il] = mean_ppvVOTING
        stdppvVOTING[ia, il] = std_ppvVOTING

np.savetxt('mean_ppv_nn.csv', meanppvNN)
np.savetxt('mean_ppv_voting.csv', meanppvVOTING)
np.savetxt('std_ppv_nn.csv', stdppvNN)
np.savetxt('std_ppv_voting.csv', stdppvVOTING)
