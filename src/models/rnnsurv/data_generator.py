'''
    Data Generator
    -----

    The data generator to be uses in training and validating
'''
import numpy as np
import pandas as pd
from tensorflow import import keras

# from utils import apply_padding, split_at_idx


class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras based on: 
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, X, y, counts, max_timesteps, padding_token, 
                 max_batch_size, min_batch_size, shuffle=True):
        'Initialization'
        self.X_dat = X
        self.y_dat = y
        self.counts = counts
        self.shuffle = shuffle
        self.max_timesteps = max_timesteps
        self.padding_token = padding_token
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        n_obs = len(self.oids)
        n_batches = int(n_obs // self.max_batch_size)
        remainder = n_obs - (n_batches * self.max_batch_size)
        if remainder >= self.min_batch_size:
            return n_batches
        else:
            return n_batches + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_oids, self.remaining_oids = \
            split_at_idx(self.remaining_oids, self.max_batch_size)

        # Generate data
        X, y = self.__data_generation(batch_oids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.oids = np.random.choice(cts.index, cts.shape[0], replace=False).tolist()
            # np.random.shuffle(self.counts.copy().index)
        else:
            self.oids = counts.index
        self.remaining_oids = self.oids.copy()

    def __data_generation(self, batch_oids):
        'Generates data containing batch_size samples' 
        # X : (n_samples, n_timesteps, n_features)
        n = len(batch_oids)
        n_features = self.X_dat.shape[1] - 1
        cts = self.counts.loc[batch_oids, :] # represents the seq lengths
        X_pad = apply_padding(self.X_dat, cts, self.max_timesteps,
                              self.padding_token, n_features)
        y_pad = apply_padding(self.y_dat, cts, self.max_timesteps,
                              self.padding_token)
        y_pad = y_pad[:, :, 0]

        # this is needed in the loss2 function
        masking = y_pad.copy()
        
        # matrix of 1's but 0's on the diagonal shape of
        # (batch_size, batch_size) will be needed in the loss function (was
        # easier to do in numpy)
        flip_i = np.abs(np.eye(n) - 1)

        seq = np.array(cts, dtype='float')        
        X_out = [X_pad, flip_i]
        y_out = [seq, masking, y_pad]

        return X_out, y_out