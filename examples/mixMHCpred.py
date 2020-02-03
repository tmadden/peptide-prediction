import pace, pace.sklearn
import sklearn.linear_model
import pprint
import random
import numpy as np
import os
from datetime import datetime


class mixMHC(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        pass

    def predict(self, samples):
        #x = [list(s.peptide) for s in samples]

        #write out for mixMHCpred. use time stamp and first peptide
        now = datetime.now()
        timestamp = str(datetime.timestamp(now))
        fname = 'predict' + timestamp + samples[0].peptide + '.txt'
        fout = 'mix' + timestamp + samples[0].peptide + '.out'
        data = open(fname, 'w')
        for s in samples:
            data.write(s.peptide)
            data.write('\n')
        data.close()

        norm_scores = np.zeros(len(samples))

        # execute mixMHCpred algorithm
        # ./MixMHCpred -i ../pace/predict.txt -o check.csv -a B4002
        # NOTE assuming only running for single allele, so using the first sample allele name below.
        d = "/home/dcraft/ImmunoOncology/MixMHCpred-master/"
        sysCommand = d + "MixMHCpred -i " + fname + " -o " + fout + " -a " + samples[
            0].allele
        print('executing mixMHCpred exec with line: ' + sysCommand)
        os.system(sysCommand)

        #process the output file
        with open(fout) as fp:
            line = fp.readline()
            cnt = 0
            while line:
                #print("Line {}: {}".format(cnt, line.strip()))
                if line[0] == '#' or line.startswith('Peptide'):
                    pass
                    #print("Line {}: {}".format(cnt, line.strip()))
                else:
                    values = line.split("\t")
                    peptide = values[0]
                    score = values[1]
                    rank = values[3]
                    norm_scores[cnt] = score
                    #print("{} {} {}".format(peptide, score, rank))
                    cnt += 1
                line = fp.readline()

        norm_scores = (norm_scores - norm_scores.min()) / (
            norm_scores.max() - norm_scores.min())

        return norm_scores.tolist()


'''
scores = pace.evaluate(
    mixMHC,
    selected_lengths=[10],
    selected_alleles=['B4002'],
    dataset=pace.data.load_dataset(95),
    nbr_train=10,  #CHANGE!! 10,1000
    nbr_test=1000,
    random_seed=1)

pprint.pprint(scores)
'''