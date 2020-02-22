'''
    Utils
    -----

    Utility functions to aid in data processing, training, etc.
'''
import os
import numpy as np
import pandas as pd


def split_at_idx(a: np.ndarray, idx: int):
    '''Splits an array into two at the index given.
    
    Example:
    x = np.arange(20.0)
    while len(x) > 0:
        x_new, x = split_at_idx(x, 3)
        print(x_new)    
    '''
    return a[:idx], a[idx:]


def apply_padding(df, survstats, time_width, padding_token=-999, n_features=None):
    '''Gets batch matrix and applies padding.
    :param df: The full raw data file, either X or y, train or test
    :param survstats: The limited dataframe with oid as the index and `tte` 
        representing the sequence length
    :param time_width: (int) the max sequence length to truncate to or add padding
    :param padding_token: the value to use to represent padding
    :param n_features: the number of features in the X (if y data, leave None)
    :returns: 3D numpy array to use in batch processing
    
    '''
    if not n_features: 
        n_features = 2   # the number of columns needed for y_batch
    
    batch = np.empty((0, time_width, n_features))
    # process observations of the same length together
    for s, val in survstats.iterrows():
        out = np.array(df.query('oid == @s').drop('oid', axis=1))
        out = np.reshape(out, newshape=(1, int(val.tte), n_features))
        out = out[:, :time_width, :]  # in case it's too long for padding
        out = np.pad(
            out, 
            pad_width=((0, 0), (0, time_width - out.shape[1]), (0, 0)), 
            mode='constant', 
            constant_values=padding_token
        )
        batch = np.append(arr=batch, values=out, axis=0)
    
    return batch


def get_data(path_to_file='data/processed/',
             X_filename='rain_X_train.csv',
             y_filename='rain_y_train.csv',
             nrows=None):
    '''Gets X and y data and makes it ready for use.'''

    X_train = pd.read_csv(os.path.join(path_to_file, X_filename),
                          nrows=nrows, header=None)
    y_train = pd.read_csv(os.path.join(path_to_file, y_filename),
                          nrows=nrows, header=None)
    X_train.rename({0: 'oid'}, axis=1, inplace=True)
    y_train.rename({0: 'oid'}, axis=1, inplace=True)

    return X_train, y_train
