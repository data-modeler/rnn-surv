# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read data
    dat = pd.read_csv(os.path.join(input_filepath, 'Aids2.csv'))
    dat.drop('Unnamed: 0', axis=1, inplace=True)

    # split X, y
    X = dat.copy()[['state', 'sex', 'T.categ', 'age']]
    y = y = pd.DataFrame({
        'tte': dat.death - dat.diag,
        'event': [1 if val == 'D' else 0 for val in dat.status]
    })

    # get feature names by type
    categorical_feature_mask = X.dtypes == object
    cat_names = X.columns[categorical_feature_mask].tolist()
    num_names = X.columns[~categorical_feature_mask].tolist()

    # split train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # get preprocessors
    ohe = OneHotEncoder(sparse=False).fit(X_train[cat_names])
    scaler = StandardScaler().fit(X_train[num_names])

    # transform train and test
    X_train = np.concatenate((
        ohe.transform(X_train[cat_names]),
        scaler.transform(X_train[num_names])),
        axis=1
    )
    X_test = np.concatenate((
        ohe.transform(X_test[cat_names]),
        scaler.transform(X_test[num_names])),
        axis=1
    )
    # save out
    np.savetxt(
        os.path.join(output_filepath, 'aids_X_train.csv'),
        X_train,
        delimiter=','
    )
    np.savetxt(
        os.path.join(output_filepath, 'aids_X_test.csv'),
        X_test,
        delimiter=','
    )
    y_train.to_csv(
        os.path.join(output_filepath, 'aids_y_train.csv'),
        header=False,
        index=False
    )
    y_test.to_csv(
        os.path.join(output_filepath, 'aids_y_test.csv'),
        header=False,
        index=False
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
