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


def predict_rnnsurv(modelname, modelpath, datapath, outpath=None):
    """ Predicts and outputs risks and probabilities over time for each new
    observation.
    """

    if outpath is None:
        outpath = datapath

    print('Getting Data...')
    xt = get_data(path_to_file=datapath, filename='rain_X_test.csv', nrows=1000)

    risk_oids = xt['oid'].drop_duplicates().astype(int)

    print("Loading Model...")
    with open(os.path.join(modelpath, f"{modelname}.json"), "r") as json_file:
        model_json = json_file.read()

    with open(os.path.join(modelpath, f"{modelname}_data_params.json"), "r") as json_file:
        params_str = json_file.read()

    params = json.loads(params_str)

    model = model_from_json(model_json)
    model.load_weights(os.path.join(modelpath, f"{modelname}.h5"))

    test_generator = DataGenerator(xt, prediction=True, **params)

    pred = model.predict(test_generator)

    risks = pd.DataFrame({
        'oid': risk_oids,
        'risk': np.transpose(pred[0])[0]
    })
    risk_out_loc = os.path.join(outpath, f"{modelname}_output_risks.csv")
    risks.to_csv(risk_out_loc, index=False)

    probs = pd.DataFrame(pred[1], index=risk_oids).reset_index(drop=False)
    probs_out_loc = os.path.join(outpath, f"{modelname}_output_probs.csv")
    probs.to_csv(probs_out_loc, index=False)

 
if __name__ == '__main__':

    MODELNAME = 'model-002'

    BASEPATH = up(up(up(__file__)))
    DATAPATH = os.path.join(BASEPATH, 'data', 'processed')
    MODELPATH = os.path.join(BASEPATH, 'models')

    predict_rnnsurv(MODELNAME, MODELPATH, DATAPATH)
