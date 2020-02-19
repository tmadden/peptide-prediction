import pace, pace.featurization
import pprint
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# Make sure the model is reproducible
from numpy.random import seed
seed(1171)
from tensorflow import set_random_seed
set_random_seed(1231)

class DonJulioBlanco(pace.PredictionAlgorithm):
    
    def __init__(self):
        self.encoding_name = 'onehot'

    ### Define Model
    def create_model_1D(self, dim_1D, n_hidden_1, dropout_rate):
        model = Sequential()
        model.add(Dense(n_hidden_1, input_dim=dim_1D, activation='relu'))
        model.add(keras.layers.Dropout(rate=dropout_rate, noise_shape=None, seed=None))
        model.add(Dense(1, activation='sigmoid')) 
        return model
    
    ### Define Callbacks
    def get_callbacks(self, patience_lr, patience_es):
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-3, mode='auto')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_es, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        return [reduce_lr_loss, early_stop]
    
    def train(self, binders, nonbinders):
        ### Data prep
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)
        
        encoded_x = pace.featurization.encode(x, self.encoding_name)
        dim_1D = encoded_x.shape[1]
        
        ### Model params
        nEpochs = 15 
        batch_size = 50 
        n_hidden_1 = 50
        dropout_rate = 0.0
        patience_lr = 2
        patience_es = 4
        callbacks = self.get_callbacks(patience_lr, patience_es)
        
        ### Train model
        model = None
        model = self.create_model_1D(dim_1D, n_hidden_1, dropout_rate)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x = encoded_x, y = y,
                verbose=0,
                batch_size=batch_size, 
                epochs=nEpochs, shuffle=True, 
                validation_split=0.1,
                class_weight=None, sample_weight=None, initial_epoch=0,
                callbacks = callbacks)
        
        self.model = model
    
    
    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = pace.featurization.encode(x, self.encoding_name)
        return self.model.predict(encoded_x).squeeze()



### Evaluate algorithm using PACE.
alleles = ['A0203', 'A6802', 'B3501']
for allele in alleles:
    eval_results = pace.evaluate(DonJulioBlanco,
                                 selected_lengths=[9], 
                                 selected_alleles=[allele], 
                                 dataset=pace.data.load_dataset(16), 
                                 nbr_train=10, nbr_test=1000)
    print(allele)
    pprint.pprint(eval_results)


