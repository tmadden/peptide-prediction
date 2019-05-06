import pace, pace.sklearn, pace.featurization
import pprint

import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from pace.custom_keras_layers import RSDLayer, Weaveconcatenate

from sklearn.preprocessing import OneHotEncoder

import pickle
import numpy as np

import csv


class StackDownNeuralNet(pace.PredictionAlgorithm):

    # for now hard coding this for 16 alleles but should pass in via init method or something
	# def __init__(self, allele_set_size):
    def __init__(self):
        self.allele_set_size = 16


    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]

        y = [1] * len(hits) + [0] * len(misses)

        # allele_list = np.unique(xa)
        
        # consider rewriting define_architecture() to take in only certain peptide lengths 
        # or other potentially variable inputs
        # note to tom/me: this next line should only be done once but as is it will be done every call.
        # but the model also needs to be "cleared" every training round. to look into...
        
        self.model = self.define_architecture(self.allele_set_size)

        # PEPTIDE ************
        ipep = self.featurize_peptides_for_fan_merge(x,aaDim=5)

        # MHC ALLELE ************
        d, indStart, indEnd = self.featurize_alleles_for_fan_merge(xa,self.allele_set_size)
        # print('DEBUG CRAFT')
        # print(indStart[5])
        # print(indEnd[5])
        m5 = d[:, indStart[5]:indEnd[5]]
        print(m5.shape)

        self.model.fit([ipep[0], ipep[1], ipep[2], ipep[3], d[:, indStart[0]:indEnd[0]],
            d[:, indStart[1]:indEnd[1]], d[:, indStart[2]:indEnd[2]], d[:, indStart[3]:indEnd[3]],
            d[:, indStart[4]:indEnd[4]], d[:, indStart[5]:indEnd[5]]], y,
            epochs=50, batch_size=500)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        xa = [s.allele for s in samples]

        # PEPTIDE ************
        ipep = self.featurize_peptides_for_fan_merge(x,aaDim=5)

        # MHC ALLELE ************
        d, indStart, indEnd = self.featurize_alleles_for_fan_merge(xa,self.allele_set_size)

        rval = self.model.predict([ipep[0], ipep[1], ipep[2], ipep[3], d[:, indStart[0]:indEnd[0]],
                                   d[:, indStart[1]:indEnd[1]], d[:, indStart[2]
                                       :indEnd[2]], d[:, indStart[3]:indEnd[3]],
                                   d[:, indStart[4]:indEnd[4]], d[:, indStart[5]:indEnd[5]]])
        # print(rval)
        # print(type(rval))
        # print(rval.shape)
        return rval[:,0].tolist()

    def define_architecture(self, allele_set_size):

        # 4 types of inputs: peptides of length 8 to 11
        pepLengths = [8, 9, 10, 11]

        # amino acid encoding dimension (5D encoding) for the peptides
        aaDim = 5

        # create the four peptide inputs, 8,9,10, and 11: the outermost layer,
        # meaning the inputs, of the fan architecture

        inputPep = []

        for i in range(len(pepLengths)):
            thename='pep'+str(i+8)    
            inputPep.append(Input(shape=(pepLengths[i]*aaDim,), dtype='float32', name=thename))


        #RSDLayer(output_length, input_num_blocks)    
        L10 = RSDLayer(aaDim*10, 11, activation='relu', name='L10')(inputPep[3])

        W20 = Weaveconcatenate([inputPep[2],L10], name='W20')

        L9 = RSDLayer(aaDim*9, 10, activation='relu', name='L9')(W20)

        W18 = Weaveconcatenate([inputPep[1],L9], name='W18')

        L8 = RSDLayer(aaDim*8, 9, activation='relu', name='L8')(W18)

        W16 = Weaveconcatenate([inputPep[0],L8], name='W16')

        L7 = RSDLayer(aaDim*7, 8, activation='relu', name='L7')(W16)

        #now we will weave this with the MHC which will also get spatially reduced down to 6 'units'.

        #output batch size once when weave with MHC
        obs=2

        shared76 = RSDLayer(obs*6, 7, activation='relu', name='pepShared76')(L7)
        
        #following code copied from fanMergeModel.py: would be good to eventually separate out as a function (prepareMHCforNN)

        # now we will weave this with the MHC which will also get spatially reduced down to 6 'units'.
        
        vedAll = pace.featurization.get_vedAll(allele_set_size)
        snpSets = pace.featurization.get_snpSets(allele_set_size)

        inputMHC = []
        compressedMHC = []

        for i in range(6):
            thename = 'mhcSet' + str(i + 1)
            print(thename)
            # get the total size for this set
            ts = 0
            for j in range(len(snpSets[i])):
                ts = ts + vedAll[snpSets[i][j] - 1]
            print(str(ts))
            inputMHC.append(Input(shape=(ts,), dtype='float32', name=thename))
            # connect each to a dense layer
            densename = 'mhcCompress' + str(i + 1)
            compressedMHC.append(
                Dense(2, activation='relu', name=densename)(inputMHC[i]))

        # now concatenate those 6 sets for weaving in with the peptide
        inputMHC6 = concatenate(
            [compressedMHC[0], compressedMHC[1], compressedMHC[2],
                compressedMHC[3], compressedMHC[4], compressedMHC[5]],
            name='MHC6')

        # NOW DO THE WEAVE

        weaved = Weaveconcatenate([shared76, inputMHC6], name='peptideMHCweave')

        # one RSD for local interactions between peptide and MHC
        # c for 'combined'
        # at this point we have 24 neurons in the layer: 2 peptides, 2 mhcs, 2 peptides, etc
        # different valid ways to reduce this with RSD. For now try:
        # assume three input blocks
        # and a total neuron count of 8 (4 and 4) for the output
        c65 = RSDLayer(8, 3, activation='relu', name='regionalInteractions')(weaved)

        # a fully connected layer for all interaction terms. making this size 5 for now.
        cf = Dense(5, activation='relu', name='denseConnections')(c65)

        # one more
        cf2 = Dense(4, activation='relu', name='denseConnections2')(cf)

        output = Dense(1, activation='sigmoid', name='output')(cf2)

        model = Model(
            inputs=[inputPep[0], inputPep[1], inputPep[2], inputPep[3], inputMHC[0],
                    inputMHC[1], inputMHC[2], inputMHC[3], inputMHC[4], inputMHC[5]],
            outputs=[output])

        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    def featurize_peptides_for_fan_merge(self,x,aaDim=5):
        ipep = []
        n = len(x)
        # try a python list of numpy 2d arrays
        # note here filling arrays with zeros. zero might not be best choice for "blank data"
        # for the neural net. keep in mind.

        ipep.append(np.zeros((n, 8 * aaDim)))
        ipep.append(np.zeros((n, 9 * aaDim)))
        ipep.append(np.zeros((n, 10 * aaDim)))
        ipep.append(np.zeros((n, 11 * aaDim)))

        x5d = pace.featurization.do_5d_encoding(x)

        # step through peptides and fill in these arrays
        for i in range(n):
            peplen = len(x[i])
            ipep[peplen - 8][i, :] = x5d[i]
        return ipep

    def featurize_alleles_for_fan_merge(self, xa, allele_set_size):
        mhc_dict = {}

        # to do move this to a separate function and only do it once, not every time train/test is called.
        f = '/home/dcraft/ImmunoOncology/hla_prot_seq_aligned_95_clean.csv'

        anames = []

        with open(f) as csvfile:
            mhcreader = csv.reader(csvfile)
            for row in mhcreader:
                # create dictionary
                mhc_dict[row[0]] = row[1:]
                # store the list of alleles
                anames.append(row[0])
        # also store the data as a numpy 2D array
        # numcols = len(row)
        # mhc_array = np.loadtxt(open(f, "rb"), delimiter=",", usecols=range(1, numcols), dtype=np.str)

        # take all 182 positions and form full data matrix
        # this will be of size n x 182
        n = len(xa)

        alleleSNPMatrix = np.empty((n, 182), dtype='c')

        for i in range(n):
            key = xa[i]
            # print(mhc_dict[key])
            alleleSNPMatrix[i, :] = mhc_dict[key]

        # now grab the SNP locations
        # get all the snp positions as one list
        snpSets = pace.featurization.get_snpSets(allele_set_size)
        vedAll = pace.featurization.get_vedAll(allele_set_size)

        s = [item for sublist in snpSets for item in sublist]
        # subtract one from each one for 0 based indexing
        sm1 = [x - 1 for x in s]

        alleleMatrixSpecialPos = alleleSNPMatrix[:, sm1]

        # d = pd.get_dummies(alleleMatrixSpecialPos)
        # pandas does the version with reduced number of one-hots, but can't find one
        # that does entire matrix at once. use sklearn for now.

        enc = OneHotEncoder()
        d = enc.fit_transform(alleleMatrixSpecialPos)
        print('d shape')
        print(d.shape)
        # now dole these out into mhcSet1, mhcSet2, ... mhcSet6

        indStart = [None] * 6
        indEnd = [None] * 6
        cnt = 0

        # i'm sure there's a one liner for this, but,  being explicit for now since i don't know python well :)
        for i in range(6):
            thename = 'mhcSet' + str(i + 1)
            print(thename)
            # get the total size for this set
            ts = 0
            for j in range(len(snpSets[i])):
                ts = ts + vedAll[snpSets[i][j] - 1]
            indStart[i] = cnt
            cnt = cnt + ts
            indEnd[i] = cnt
            print(str(ts))

        return d, indStart, indEnd

           
scores = pace.evaluate(StackDownNeuralNet,
                    selected_lengths=[8,9,10,11],dataset=pace.data.load_dataset(16))
pprint.pprint(scores)
