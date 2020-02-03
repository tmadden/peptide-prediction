from DonJulioNoir import DJRSD, DJRSDfmln
from voting_fmln import voting_fmln
import numpy as np
import logging
import pace
import pprint
import multiprocessing
from pkg_resources import resource_stream

logging.basicConfig(level=logging.INFO)

#in matlab using similarity of logo plots I made neightbor lists. see logoSimilarities.m
#using a cutoff of 1, not all alleles have neighbors, but most do
nd = {'A0101': ['A3601']}
nd.update({'A0201': ['A0202', 'A0203']})
nd.update({'A0202': ['A0201', 'A0204']})
nd.update({'A0203': ['A0201']})
nd.update({'A0204': ['A0202']})
nd.update({'A0207': ['A0211']})
nd.update({'A0211': ['A0207']})
nd.update({'A1101': ['A1102', 'A3402']})
nd.update({'A1102': ['A1101']})
nd.update({'A2301': ['A2402', 'A2407']})
nd.update({'A2402': ['A2301']})
nd.update({'A2407': ['A2301']})
nd.update({'A2501': ['A2601']})
nd.update({'A2601': ['A2501']})
nd.update({'A2902': ['A3002']})
nd.update({'A3002': ['A2902']})
nd.update({'A3101': ['A3303', 'A7401']})
nd.update({'A3301': ['A3303']})
nd.update({'A3303': ['A3101', 'A3301', 'A6601', 'A6801']})
nd.update({'A3401': ['A6601']})
nd.update({'A3402': ['A1101']})
nd.update({'A3601': ['A0101']})
nd.update({'A6601': ['A3303', 'A3401', 'A6801']})
nd.update({'A6801': ['A3303', 'A6601']})
nd.update({'A7401': ['A3101']})
nd.update({'B0702': ['B0704', 'B3503', 'B4201']})
nd.update({'B0704': ['B0702', 'B3503', 'B4201']})
nd.update({'B1302': ['B5201']})
nd.update({'B1501': ['B1502']})
nd.update({'B1502': ['B1501']})
nd.update({'B1801': ['B4403']})
nd.update({'B3501': ['B3507', 'B5301']})
nd.update({'B3503': ['B0702', 'B0704', 'B4201']})
nd.update({'B3507': ['B3501']})
nd.update({'B3801': ['B3802']})
nd.update({'B3802': ['B3801']})
nd.update({'B4006': ['B4501', 'B5001']})
nd.update({'B4201': ['B0702', 'B0704', 'B3503']})
nd.update({'B4402': ['B4403']})
nd.update({'B4403': ['B1801', 'B4402']})
nd.update({'B4501': ['B4006']})
nd.update({'B4601': ['C1601']})
nd.update({'B5001': ['B4006']})
nd.update({'B5201': ['B1302']})
nd.update({'B5301': ['B3501']})
nd.update({'B5401': ['B5501', 'B5601']})
nd.update({'B5501': ['B5401', 'B5502', 'B5601']})
nd.update({'B5502': ['B5501', 'B5601']})
nd.update({'B5601': ['B5401', 'B5501', 'B5502']})
nd.update({'B5701': ['B5801']})
nd.update({'B5801': ['B5701']})
nd.update({'C0202': ['C0302', 'C1202', 'C1203', 'C1601']})
nd.update({'C0302': ['C0202', 'C1202', 'C1203', 'C1601']})
nd.update({'C0303': ['C0304', 'C0801']})
nd.update({'C0304': ['C0303', 'C0801']})
nd.update({'C0403': ['C0501']})
nd.update({'C0501': ['C0403', 'C0802']})
nd.update({'C0801': ['C0303', 'C0304']})
nd.update({'C0802': ['C0501']})
nd.update({'C1202': ['C0202', 'C0302', 'C1203', 'C1601']})
nd.update({'C1203': ['C0202', 'C0302', 'C1202', 'C1601']})
nd.update({'C1402': ['C1403']})
nd.update({'C1403': ['C1402']})
nd.update({'C1601': ['B4601', 'C0202', 'C0302', 'C1202', 'C1203']})
nd.update({'G0101': ['G0103', 'G0104']})
nd.update({'G0103': ['G0101', 'G0104']})
nd.update({'G0104': ['G0101', 'G0103']})
'''
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
'''


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

    return_dict[rseed] = scoresDJRSD
    #return_dict[rseed + 100] = scoresVOTING


flists = [[8, 9], [8, 9, 10], [9, 10, 11], [10, 11]]

#choose the set of random seeds
rseeds = range(10)

manager = multiprocessing.Manager()

import pace.data
alleles = list(
    pace.data.read_alleles_file(
        resource_stream("pace", "data/alleles_95.txt")))

lengths = [8, 9, 10, 11]

meanppvNN = np.zeros(shape=(len(alleles), len(lengths)))
stdppvNN = np.zeros(shape=(len(alleles), len(lengths)))
meanppvVOTING = np.zeros(shape=(len(alleles), len(lengths)))
stdppvVOTING = np.zeros(shape=(len(alleles), len(lengths)))

for ia in range(len(alleles)):
    #form training alleles list for this allele:
    if nd.get(alleles[ia]) is None:
        my_training_alleles = [alleles[ia]]
    else:
        my_training_alleles = [alleles[ia]] + nd.get(alleles[ia])

    print('running allele ' + alleles[ia])
    print('with training alleles ' + str(my_training_alleles))

    for il in range(len(lengths)):
        m = 4
        n = lengths[il] - m

        return_dict = manager.dict()
        jobs = []

        print('length ' + str(lengths[il]) + ' with training lengths:')
        print(flists[il])
        #run jobs in parallel
        for rs in rseeds:
            p = multiprocessing.Process(
                target=worker,
                args=([alleles[ia]], my_training_alleles, [lengths[il]],
                      flists[il], m, n, rs, return_dict))
            jobs.append(p)
            p.start()

        #wait for all runs to finish:
        for proc in jobs:
            proc.join()

        ppv_valuesNN = []
        #ppv_valuesVOTING = []

        for r in return_dict.keys():
            s = return_dict[r]
            if r < 90:
                ppv_valuesNN.append(s['overall']['ppv'])
            else:
                #ppv_valuesVOTING.append(s['overall']['ppv'])
                pass

        print("allele " + alleles[ia] + ", length " + str(lengths[il]))

        mean_ppvNN = np.mean(ppv_valuesNN)
        std_ppvNN = np.std(ppv_valuesNN)
        print("  Mean ppv NN is {:.2f}".format(mean_ppvNN))
        print("  Stdev of ppv NN is {:.3f}".format(std_ppvNN))
        meanppvNN[ia, il] = mean_ppvNN
        stdppvNN[ia, il] = std_ppvNN
        '''
        mean_ppvVOTING = np.mean(ppv_valuesVOTING)
        std_ppvVOTING = np.std(ppv_valuesVOTING)
        print("  Mean ppv VOTING is {:.2f}".format(mean_ppvVOTING))
        print("  Stdev of ppv VOTING is {:.3f}".format(std_ppvVOTING))
        meanppvVOTING[ia, il] = mean_ppvVOTING
        stdppvVOTING[ia, il] = std_ppvVOTING
        '''
    np.savetxt('mean_ppv_nn_bothALONGTHEWAY.csv', meanppvNN)

    #np.savetxt('mean_ppv_voting_neighbor_allelesALONGTHEWAY.csv',
    #           meanppvVOTING)

np.savetxt('mean_ppv_nn_both.csv', meanppvNN)
#np.savetxt('mean_ppv_voting_neighbor_alleles.csv', meanppvVOTING)
np.savetxt('std_ppv_nn_both.csv', stdppvNN)
#np.savetxt('std_ppv_voting_neighbor_alleles.csv', stdppvVOTING)
