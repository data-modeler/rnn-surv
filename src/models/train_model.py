'''
    TODO:
    4. calculate C-Index from final predictions
    5. set up hyperparam tuning
'''

import os
import json
import numpy as np
import pandas as pd
from os.path import dirname as up
from src.models.rnnsurv import get_data, DataGenerator, create_model


def train_model(train_dat: tuple, val_dat: tuple, modelname: str, modelpath: str,
                datapath: str, n_epochs: int,  max_batch_size: int=128,
                min_batch_size: int=32, shuffle: bool=True,
                dense_sizes: tuple=(20, 10), lstm_sizes: tuple=(30, 30),
                dropout_prob: float=0.5, max_length: int=200,
                pad_token: int=-999, optimizer: str='adam',
                y_hat_weight: float=1.0, r_out_weight: float=1.0):
    """Creates and trains an RNN SURV model."""

    xt, yt = train_dat
    xv, yv = val_dat

    n_features = xt.shape[1] - 1 

    model_params = {
        'n_features': n_features,
        'dense_sizes': dense_sizes,
        'lstm_sizes': lstm_sizes,
        'dropout_prob': dropout_prob,
        'max_length': max_length,
        'pad_token': pad_token,
        'optimizer': optimizer,
        'loss_weights': {"y_hat": y_hat_weight, "r_out": r_out_weight}
    }

    model = create_model(**model_params)

    data_params = {
        'max_timesteps': max_length,
        'padding_token': pad_token,
        'max_batch_size': max_batch_size,
        'min_batch_size': min_batch_size,
        'shuffle': shuffle,
    }

    train_generator = DataGenerator(xt, yt, **data_params)

    val_generator = DataGenerator(\
        xv, yv, max_timesteps=max_length, padding_token=pad_token,
        max_batch_size=1024, min_batch_size=1024, shuffle=True, validation=True
    )

    model.fit(train_generator, validation_data=val_generator, epochs=n_epochs)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(modelpath, f"{modelname}.json"), "w") as json_file:
        json_file.write(model_json)

    with open(os.path.join(modelpath, f"{modelname}_data_params.json"), "w")\
        as json_file:
        json_file.write(json.dumps(data_params))

    # serialize weights to HDF5
    model.save_weights(os.path.join(modelpath, f"{modelname}.h5"))


if __name__ == '__main__':

    MODELNAME = 'model-002'

    BASEPATH = up(up(up(__file__)))
    DATAPATH = os.path.join(BASEPATH, 'data', 'processed')
    MODELPATH = os.path.join(BASEPATH, 'models')

    print('Getting Data...')
    XT = get_data(path_to_file=DATAPATH, filename='rain_X_train.csv', nrows=20000)
    YT = get_data(path_to_file=DATAPATH, filename='rain_y_train.csv', nrows=20000)
    XV = get_data(path_to_file=DATAPATH, filename='rain_X_val.csv', nrows=100000)
    YV = get_data(path_to_file=DATAPATH, filename='rain_y_val.csv', nrows=100000)

    print('Training Model...')
    train_model((XT, YT), (XV, YV), MODELNAME, MODELPATH,
                DATAPATH, n_epochs=9)
