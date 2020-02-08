'''
    4. calculate C-Index from final predictions
    5. set up hyperparam tuning
'''

import numpy as np
import pandas as pd

from rnnsurv import get_data, DataGenerator, create_model


XT, YT = get_data()

N_FEATURES = XT.shape[1] - 1 

MODEL_PARAMS = {
    'dense_sizes': (20, 10),
    'lstm_sizes': (30, 30),
    'dropout_prob': 0.5,
    'max_length': 200,
    'pad_token': -999,
    'optimizer': 'adam',
    'loss_weights': {"y_hat": 1.0, "r_out": 1.0}
}

MODEL = create_model(N_FEATURES, **MODEL_PARAMS)

PARAMS = {
    'max_timesteps': MODEL_PARAMS['max_length'],
    'padding_token': MODEL_PARAMS['pad_token'],
    'max_batch_size': 64,
    'min_batch_size': 32,
    'shuffle': True,
}

TRAIN_GENERATOR = DataGenerator(XT, YT, **PARAMS)

MODEL.fit(TRAIN_GENERATOR, epochs=9)


