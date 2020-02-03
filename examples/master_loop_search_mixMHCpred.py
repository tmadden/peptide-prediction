from mixMHCpred import mixMHC
import numpy as np
import logging
import pace
import pprint
import multiprocessing
from pkg_resources import resource_stream

logging.basicConfig(level=logging.INFO)


#wrapper function for pace.evaluate call:
def worker(test_allele, train_alleles, test_length, train_lengths, rseed,
           return_dict):
    scoresMIX = pace.evaluate(
        mixMHC,
        selected_lengths=train_lengths,
        selected_alleles=train_alleles,
        test_alleles=test_allele,
        test_lengths=test_length,
        dataset=pace.data.load_dataset(95),
        nbr_train=10,
        nbr_test=1000,
        random_seed=rseed)

    return_dict[rseed] = scoresMIX


#choose the set of random seeds
rseeds = range(10)

manager = multiprocessing.Manager()
'''
import pace.data
alleles = list(
    pace.data.read_alleles_file(
        resource_stream("pace", "data/alleles_95.txt")))
'''
#just read alleles the mixMHCpred is trained for.
with open('/home/dcraft/ImmunoOncology/inboth.txt') as f:
    content = f.readlines()
alleles = [x.strip() for x in content]

lengths = [8, 9, 10, 11]

meanppvMIX = np.zeros(shape=(len(alleles), len(lengths)))
stdppvMIX = np.zeros(shape=(len(alleles), len(lengths)))

for ia in range(len(alleles)):

    print('running allele ' + alleles[ia])

    for il in range(len(lengths)):

        return_dict = manager.dict()
        jobs = []

        #run jobs in parallel
        for rs in rseeds:
            p = multiprocessing.Process(
                target=worker,
                args=([alleles[ia]], [alleles[ia]], [lengths[il]],
                      [lengths[il]], rs, return_dict))
            jobs.append(p)
            p.start()

        #wait for all runs to finish:
        for proc in jobs:
            proc.join()

        ppv_valuesMIX = []

        for r in return_dict.keys():
            s = return_dict[r]
            ppv_valuesMIX.append(s['overall']['ppv'])

        print("allele " + alleles[ia] + ", length " + str(lengths[il]))

        mean_ppvMIX = np.mean(ppv_valuesMIX)
        std_ppvMIX = np.std(ppv_valuesMIX)
        print("  Mean ppv MIX is {:.2f}".format(mean_ppvMIX))
        print("  Stdev of ppv MIX is {:.3f}".format(std_ppvMIX))
        meanppvMIX[ia, il] = mean_ppvMIX
        stdppvMIX[ia, il] = std_ppvMIX

    np.savetxt('mean_ppv_mixmhcALONGTHEWAY.csv', meanppvMIX)

np.savetxt('mean_ppv_mixmhc.csv', meanppvMIX)
np.savetxt('std_ppv_mixmhc.csv', stdppvMIX)
