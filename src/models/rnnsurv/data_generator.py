'''
    Data Generator
    -----

    The data generator to be uses in training and validating
'''
import numpy as np
import pandas as pd
from tensorflow import keras

from utils import apply_padding, split_at_idx

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras
    based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, X, y, max_timesteps, padding_token, 
                 max_batch_size, min_batch_size, shuffle=True):
        'Initialization'
        assert(min_batch_size > 1), 'You must have a `min_batch_size` greater than 1'
        self.X_dat = X
        self.y_dat = y
        self._make_survstats()
        self.shuffle = shuffle
        self.max_timesteps = max_timesteps
        self.padding_token = padding_token
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.on_epoch_end()

    def _make_survstats(self):
        '''Constructs dataframe that has the traditional Survival Analysis
        dependent variable with event (1 or 0) and time-to-event (or sequence
        length)
        '''
        df = self.y_dat.copy().groupby('oid').max()
        df.index = df.index.astype('float')
        df.columns = ['event', 'tte']

        e_dist = df.event.value_counts()
        df['e_freq'] =\
            df.event.map(e_dist) / len(df)

        t_dist = df.tte.value_counts()
        df['t_freq'] =\
            df.tte.map(t_dist) / len(df)

        df['weight'] = df.e_freq * df.t_freq

        self.survstats = df
         

    def __len__(self):
        'Denotes the number of batches per epoch'
        n_obs = len(self.oids)
        n_batches = int(n_obs // self.max_batch_size)
        remainder = n_obs - (n_batches * self.max_batch_size)
        if remainder >= self.min_batch_size:
            return n_batches + 1
        else:
            return n_batches


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
            self.oids = np.random.choice(self.survstats.index,
                                         self.survstats.shape[0],
                                         replace=False).tolist()
        else:
            self.oids = self.survstats.index
        self.remaining_oids = self.oids.copy()


    def __data_generation(self, batch_oids):
        'Generates data containing batch_size samples' # X : (n_samples, n_timesteps, n_features)
        n = len(batch_oids)
        n_features = self.X_dat.shape[1] - 1
        batch_cts = self.survstats.loc[batch_oids, :] # represents the seq lengths
        X_pad = apply_padding(self.X_dat, batch_cts, self.max_timesteps, self.padding_token, n_features)
        y_pad = apply_padding(self.y_dat, batch_cts, self.max_timesteps, self.padding_token)
        y_pad = y_pad[:, :, 0]

        # this is needed in the loss2 function
        masking = y_pad.copy()
        
        # matrix of 1's but 0's on the diagonal shape of (batch_size, batch_size) will
        # be needed in the loss function (was easier to do in numpy)
        flip_i = np.abs(np.eye(n) - 1)

        seq = np.array(batch_cts, dtype='float')        
        X_out = [X_pad, masking, flip_i]
        y_out = [seq, y_pad]

        return X_out, y_out


if __name__ == '__main__':
    # demonstrate usage by printing the shapes of the output
    X_train = pd.read_csv('../../../data/processed/rain_X_train.csv',
                          nrows=2000, header=None)
    y_train = pd.read_csv('../../../data/processed/rain_y_train.csv',
                          nrows=2000, header=None)
    X_train.rename({0: 'oid'}, axis=1, inplace=True)
    y_train.rename({0: 'oid'}, axis=1, inplace=True)

    test_params = {
        'max_timesteps': 300, 
        'padding_token': -999,
        'max_batch_size': 3, 
        'min_batch_size': 2, 
        'shuffle': True
    }

    training_generator = DataGenerator(X_train, y_train, **test_params)

    for i in training_generator:
        print(i[0][0].shape, i[0][1].shape, i[0][2].shape,
              i[1][0].shape, i[1][1].shape)
