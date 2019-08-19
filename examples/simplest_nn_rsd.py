import pace, pace.sklearn, pace.featurization
import pprint

import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from custom_keras_layers import RSDLayer
from sklearn.preprocessing import OneHotEncoder

import pickle
import numpy as np

import csv


class SimpleRSDNeuralNet(pace.PredictionAlgorithm):

    # for now hard coding this for 16 alleles but should pass in via init method or something
	# def __init__(self, allele_set_size):
    def __init__(self):
        self.allele_set_size = 16


    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        
        y = [1] * len(hits) + [0] * len(misses)
  
        self.model = self.define_architecture(self.allele_set_size)

        # PEPTIDE ************
        ipep = self.featurize_peptides_simple(x,aaDim=0)

        self.model.fit([ipep], y,
            epochs=100, batch_size=60)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        
        # PEPTIDE ************
        ipep = self.featurize_peptides_simple(x,aaDim=0)

        rval = self.model.predict(ipep)
        # print(rval)
        # print(type(rval))
        # print(rval.shape)
        return rval[:,0].tolist()

    def define_architecture(self, allele_set_size):

        # amino acid encoding dimension (5D encoding) for the peptides
        aaDim = 20 #hard coded 20 is for one hot
        pep_length = 9

        my_input = Input(shape=(pep_length*aaDim,), dtype='float32', name='pep9input')
        
        #RSDLayer(output_length, input_num_blocks)
        #RSD, now that I fixed the bug where I wasn't incrementing seqPos, does well...looks better than Dense.
        cf = RSDLayer(6*8, 9, activation='relu', name='L8')(my_input)
        #cf = Dense(50, activation='relu', name='denseConnections')(my_input)

        output = Dense(1, activation='sigmoid', name='output')(cf)

        model = Model(inputs=my_input, outputs=output)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    def featurize_peptides_simple(self,x,aaDim=5):

        if aaDim==5:
            n = len(x)
            ipep = np.zeros((n, 9 * aaDim))

            x5d = pace.featurization.do_5d_encoding(x)

            # step through peptides and fill in these arrays
            for i in range(n):
                ipep[i, :] = x5d[i]
        else:
            encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
            encoder.fit(x)
            ipep = encoder.transform(x).toarray()
        return ipep

import logging
logging.basicConfig(level=logging.INFO)

#shortcuts:
a02 = ['A0203']
a68 = ['A6802']
b35 = ['B3501']

import multiprocessing


#wrapper function for pace.evaluate call:
def worker(rseed, return_dict):
    scores = pace.evaluate(
        SimpleRSDNeuralNet,
        selected_lengths=[9],
        selected_alleles=b35,
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
print('Stdev of ppv is {:.2f}'.format(std_ppv))
