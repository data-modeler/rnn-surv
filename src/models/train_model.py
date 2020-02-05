'''
# TODO:
    1. create a generator to produce batches
     - the batches will include observations with the same number of timesteps
       up to a max batch size
    2. assemble the sequential model with 
     - TimeDistributed(Dense(n_hidden), input_size(None, n_dimensions) x 2
     - LSTM x 2
    3. calculate losses
    4. calculate C-Index from final predictions
    5. set up hyperparam tuning
'''

import numpy as np
import pandas as pd

dat = pd.read_csv('../../data/processed/rain_X_train.csv', nrows=20000)

print(dat.columns)

