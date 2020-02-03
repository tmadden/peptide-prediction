import pace, pace.sklearn
import sklearn.linear_model
import pprint

import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils

from custom_keras_layers import RSDLayer, Weaveconcatenate
from encoder import fmln_plus


class pan_length_rsd(pace.PredictionAlgorithm):
    def __init__(self, encoding_style='one_hot'):
        self.encoding_style = encoding_style
        self.dim_dict = {'one_hot': 20, '5d': 5}

    def create_model(self):

        # 4 types of inputs: peptides of length 8 to 11
        pepLengths = [8, 9, 10, 11]

        # amino acid encoding dimension (5D encoding) for the peptides

        aaDim = self.dim_dict[self.encoding_style]
        inputPep = []

        # WARNING MIGHT NEED TO CHANGE dtype below for one_hot encoding!
        for i in range(len(pepLengths)):
            thename = 'pep' + str(i + 8)
            inputPep.append(
                Input(
                    shape=(pepLengths[i] * aaDim, ),
                    dtype='float32',
                    name=thename))

        #RSDLayer(output_length, input_num_blocks)
        L10 = RSDLayer(
            aaDim * 10, 11, activation='relu', name='L10')(inputPep[3])
        W20 = Weaveconcatenate([inputPep[2], L10], name='W20')
        L9 = RSDLayer(aaDim * 9, 10, activation='relu', name='L9')(W20)
        W18 = Weaveconcatenate([inputPep[1], L9], name='W18')
        L8 = RSDLayer(aaDim * 8, 9, activation='relu', name='L8')(W18)
        W16 = Weaveconcatenate([inputPep[0], L8], name='W16')
        #note next one, for L7, sdsize does not need to be aaDim.
        sdsize = 7
        L7 = RSDLayer(sdsize * 7, 8, activation='relu', name='L7')(W16)
        #output size
        '''
        obs = 2
        shared76 = RSDLayer(
            obs * 6, 7, activation='relu', name='pepShared76')(L7)
        cf2 = Dense(4, activation='relu', name='denseConnections2')(L7)
        '''
        output = Dense(1, activation='sigmoid', name='output')(L7)

        model = Model(
            inputs=[inputPep[0], inputPep[1], inputPep[2], inputPep[3]],
            outputs=[output])

        return model

    ### Define Callbacks
    def get_callbacks(self, patience_lr, patience_es):
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=patience_lr,
            verbose=1,
            min_delta=1e-3,
            mode='auto')
        early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience_es,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
        return [reduce_lr_loss, early_stop]

    def featurize_peptides_for_fan_merge(self, x, aaDim=5):
        ipep = []
        n = len(x)
        # try a python list of numpy 2d arrays
        # note here filling arrays with zeros. zero might not be best choice for "blank data"
        # for the neural net. keep in mind.

        ipep.append(np.zeros((n, 8 * aaDim)))
        ipep.append(np.zeros((n, 9 * aaDim)))
        ipep.append(np.zeros((n, 10 * aaDim)))
        ipep.append(np.zeros((n, 11 * aaDim)))

        if self.encoding_style == '5d':
            xenc = pace.featurization.do_5d_encoding(x)
        else:
            '''
            encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
            encoder.fit(x)
            xenc = encoder.transform(x).toarray()
            '''
            xenc = pace.featurization.do_variable_length_one_hot_encoding(x)

        # step through peptides and fill in these arrays
        for i in range(n):
            peplen = len(x[i])
            ipep[peplen - 8][i, :] = xenc[i]
        return ipep

    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]

        y = [1] * len(hits) + [0] * len(misses)

        ipep = self.featurize_peptides_for_fan_merge(
            x, aaDim=self.dim_dict[self.encoding_style])

        ### Model params
        nEpochs = 60
        batch_size = 50
        patience_lr = 2
        patience_es = 4
        callbacks = self.get_callbacks(patience_lr, patience_es)

        ### Train model
        model = None

        model = self.create_model()
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        model.fit([ipep[0], ipep[1], ipep[2], ipep[3]],
                  y,
                  verbose=0,
                  batch_size=batch_size,
                  epochs=nEpochs,
                  shuffle=True,
                  validation_split=0.1,
                  class_weight=None,
                  sample_weight=None,
                  initial_epoch=0,
                  callbacks=callbacks)

        self.model = model

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        xa = [s.allele for s in samples]

        # PEPTIDE ************
        ipep = self.featurize_peptides_for_fan_merge(
            x, aaDim=self.dim_dict[self.encoding_style])

        rval = self.model.predict([ipep[0], ipep[1], ipep[2], ipep[3]])
        return rval[:, 0].tolist()
