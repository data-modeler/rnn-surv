'''
    Utils
    -----

    Utility functions to aid in data processing, training, etc.
'''
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


def apply_padding(df, counts_df, time_width, padding_token=-999, n_features=None):
    '''Gets batch matrix and applies padding.
    :param df: The full raw data file, either X or y, train or test
    :param counts_df: The limited dataframe with oid as the index and `counts` 
        representing the sequence length
    :param time_width: (int) the max sequence length to truncate to or add padding
    :param padding_token: the value to use to represent padding
    :param n_features: the number of features in the X (if y data, leave None)
    :returns: 3D numpy array to use in batch processing
    
    '''
    if not n_features: 
        n_features = 2   # the number of columns needed for y_batch
    
    batch = np.empty((0, time_width, n_features))
    for s, val in counts_df.iterrows():
        out = np.array(df.query('oid == @s').drop('oid', axis=1))
        out = np.reshape(out, newshape=(1, val.counts, n_features))
        out = out[:, :time_width, :]  # in case it's too long for padding
        out = np.pad(
            out, 
            pad_width=((0, 0), (0, time_width - out.shape[1]), (0, 0)), 
            mode='constant', 
            constant_values=padding_token
        )
        batch = np.append(arr=batch, values=out, axis=0)
    
    return batch


