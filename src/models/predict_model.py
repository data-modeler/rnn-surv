"""
    Predict RNN-SURV model
"""

import os
import numpy as np
import pandas as pd
import json
from os.path import dirname as up
from tensorflow.keras.models import model_from_json
from src.models.rnnsurv import get_data, DataGenerator, create_model

MODELNAME = 'model-002'

BASEPATH = up(up(up(__file__)))
DATAPATH = os.path.join(BASEPATH, 'data', 'processed')
MODELPATH = os.path.join(BASEPATH, 'models')

print('Getting Data...')
# XT, YT = get_data(path_to_file=DATAPATH, X_filename='rain_X_test.csv',
#                   y_filename='rain_y_test.csv', nrows=None)


print("Loading Model...")
with open(os.path.join(MODELPATH, f"{MODELNAME}.json"), "r") as json_file:
    MODEL_JSON = json_file.read()

with open(os.path.join(MODELPATH, f"{MODELNAME}_data_params.json"), "r") as json_file:
    PARAMS = json_file.read()

# MODEL = model_from_json(MODEL_JSON)
# MODEL.load_weights(os.path.join(MODELPATH, f"{MODELNAME}.h5"))

with open(os.path.join(MODELPATH, f"{MODELNAME}_data_params.json"), "w") as json_file:
    json_file.write(json.dumps(PARAMS))

# VAL_GENERATOR = DataGenerator(XV, YV, validation=True, **PARAMS)

