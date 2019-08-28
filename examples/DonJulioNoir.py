import pace, pace.sklearn
import sklearn.linear_model
import pprint

import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils

from custom_keras_layers import RSDLayer

# Make sure the model is reproducible
#from numpy.random import seed
#seed(1171)
#from tensorflow import set_random_seed
#set_random_seed(1231)


class DJRSD(pace.PredictionAlgorithm):
    ### Define Model
    def create_model_1D(self, dim_1D, n_hidden_1, dropout_rate):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='relu'))
        #model.add(
        #    keras.layers.Dropout(
        #        rate=dropout_rate, noise_shape=None, seed=None))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def create_model_RSD(self, aa_dim, peptide_length, n_hidden_1):
        # RSD is regional step down layer. Like dense but only 'local' connections
        # note for this RSD, n_hidden_1 must be divisible by peptide_length - 1.
        my_input = Input(
            shape=(aa_dim * peptide_length, ),
            dtype='float32',
            name='pep9input')
        cf = RSDLayer(
            n_hidden_1, peptide_length, activation='relu', name='L8')(my_input)
        output = Dense(1, activation='sigmoid', name='output')(cf)
        model = Model(inputs=my_input, outputs=output)

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

    def train(self, binders, nonbinders):
        ### Data prep
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()
        dim_1D = len(encoder.categories_) * 20

        ### Model params
        nEpochs = 15
        batch_size = 50
        dropout_rate = 0.0
        patience_lr = 2
        patience_es = 4
        callbacks = self.get_callbacks(patience_lr, patience_es)

        ### Train model
        model = None
        #model = self.create_model_1D(dim_1D, n_hidden_1, dropout_rate)
        aa_dim = 20
        peptide_length = len(encoder.categories_)
        n_hidden_1 = (peptide_length - 1) * 6

        model = self.create_model_RSD(aa_dim, peptide_length, n_hidden_1)
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        model.fit(
            x=encoded_x,
            y=y,
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

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.model.predict(encoded_x).squeeze()