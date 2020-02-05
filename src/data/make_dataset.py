# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DataFrame = pd.core.frame.DataFrame

def save_files(path:str, filename:str, dat, **kwargs):
    """ Saves file based on criteria.

    :param path: The location in which to save the file.
    :param filename: The name of the file to save.
    :param dat: The data to be saved, either a pandas DataFrame or numpy array.
    :param dattype: The type of the data, either 'np' or 'df'. Defaults to 'np'.
    """
        
    try:
        os.makedirs(os.path.join('data', path))
    except FileExistsError:
        pass

    if isinstance(dat, DataFrame):
        dat.to_csv(os.path.join('data', path, filename), index=False, **kwargs)
    else:
        np.savetxt(os.path.join('data', path, filename), dat, delimiter=',')


def process_weather_station(stat:DataFrame) -> DataFrame:
    """ Processes the Irish weather data to make a censored time-to-event 
        prediction model.

    :param stat: Weather data for only one station

    :returns: Processed data for the station, ready to be appended.
    :rtype: pandas DataFrame
    """
    stat['prev_rain_flag'] = np.append(np.nan, stat.rain_flag)[0:-1]

    # remove consecutive rains
    stat['rem'] = 0
    stat.loc[(stat.rain_flag == 1) & (stat.prev_rain_flag == 1), 'rem'] = 1
    stat = stat\
        .query('rem == 0')\
        .drop('rem', axis=1)\
        .reset_index(drop=True)

    # remove observations before first start
    stat['start'] = 0
    stat.loc[(stat.rain_flag == 0) & (stat.prev_rain_flag == 1), 'start'] = 1
    first_start = stat.query('start == 1').index.tolist()[0]
    stat = stat.loc[first_start:, ].reset_index(drop=True)

    # create sequence variable and observation id
    sequence = []
    obs_id = []
    prev_oid = 0
    for i in stat.index:
        if stat.loc[i, 'start'] == 1:
            val = 1
            prev = 1
            oid = prev_oid + 1
            prev_oid = oid 
        else:
            val = 1 + val
            prev = val
            oid = prev_oid

        sequence.append(val)
        obs_id.append(oid)

    stat['seq'] = sequence
    stat['oid'] = obs_id

    return stat.drop(['prev_rain_flag', 'start'], axis=1)


def process_interim_weather_data(input_path, output_path):
    """ Processes the Irish weather data to make a censored time-to-event 
        prediction model.
    """
    print('Reading raw data...')
    dat = pd.read_csv(os.path.join(input_path, 'hourly_irish_weather.csv'))

    dat.drop('Unnamed: 0', axis=1, inplace=True)

    # create event flag
    dat['rain_flag'] = 0
    dat.loc[dat.rain != 0, 'rain_flag'] = 1

    # it's easier to process separately for each station
    stations = dat.station.unique().tolist()
    list_of_df = [
        process_weather_station(dat.copy().query('station == @station'))
        for station in tqdm(stations, desc="Processing stations: ")
    ]
    out = pd.concat(list_of_df, ignore_index=True)

    stations_dict = {station: i*100000 for i, station in enumerate(stations)}
    dat.oid = dat.station.map(stations_dict) + dat.oid

    print('Saving interim File...')
    save_files('interim', 'rain_interim.csv', out)


def make_survival_data(data):
    # delete some data to make it a survival analysis problem
    rules = data\
        .loc[:, ['station', 'oid', 'seq']]\
        .groupby(['station', 'oid'])\
        .max()\
        .sample(frac=0.5)\
        .reset_index()\
        .copy()

    rules['keep'] = (np.random.random(rules.shape[0]) * rules.seq).round(0)
    # because 0 would effectively delete the observation
    rules.loc[rules.keep == 0.0, 'keep'] = 1.0

    out = pd.merge(
        data, rules, how='left', on=['station', 'oid'], suffixes=('', '_y')
    )\
        .fillna(value={'keep': 10000000})\
        .query('seq <= keep')\
        .drop(['seq_y', 'keep', 'county', 'longitude', 'latitude', 'index',
               'rain'], axis=1)
    return out


def split_survival_data(data, frac=0.7):
    train_ids = data.query("seq == 1").sample(frac=frac).oid.tolist()
    
    train_dat = data.query("oid in @train_ids")
    test_dat = data.query("oid not in @train_ids")
              
    x_cols = ['station', 'temp', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl',
              'wdsp', 'wddir', 'month', 'oid']
    y_cols = ['oid', 'rain_flag', 'seq']

    return train_dat[x_cols], test_dat[x_cols], train_dat[y_cols], test_dat[y_cols]


def process_final_weather_data():
    """ Reads the interim data after exploring in Jupyter notebook and
        implements the final processing steps
    """
    print('Reading interim data...')
    out = pd.read_csv(os.path.join('data', 'interim', 'rain_interim.csv'))
    stations = out.station.unique().tolist()

    print('Processing final data...')
    # remove the outlier
    out = out\
        .query("(station != 'Roches_Point') & (oid != 4277)")\
        .reset_index(drop=True)

    out = make_survival_data(out)

    # make oid unique
    add_to_oid = {station: i * 100000 for i, station in enumerate(stations)}
    out['oid'] = out.oid + out.station.map(add_to_oid)

    # use month of the year as a predictor
    out['month'] = pd.DatetimeIndex(out.date.to_numpy(np.datetime64)).month

    X_train, X_test, y_train, y_test = split_survival_data(out)
    
    # get preprocessors
    cats = ['station', 'month']
    nums = [col for col in X_train.columns if col not in cats + ['oid']]
    imputer = SimpleImputer().fit(X_train[nums])
    scaler = StandardScaler().fit(X_train[nums])
    ohe = OneHotEncoder(sparse=False).fit(X_train[cats])

    # transform train and test
    X_train = np.concatenate((
        X_train[['oid']],
        ohe.transform(X_train[cats]),
        scaler.transform(imputer.transform(X_train[nums]))),
        axis=1
    )
    X_test = np.concatenate((
        X_test[['oid']],
        ohe.transform(X_test[cats]),
        scaler.transform(imputer.transform(X_test[nums]))),
        axis=1
    )

    print('Saving final data files...')
    save_files('processed', 'rain_X_train.csv', X_train)
    save_files('processed', 'rain_X_test.csv', X_test)
    save_files('processed', 'rain_y_train.csv', y_train, header=False)
    save_files('processed', 'rain_y_test.csv', y_test, header=False)
    

def process_aids_data(input_path, output_path):
    """ Processes the Aids2 data for analysis in R and Python. """

    # read data
    dat = pd.read_csv(os.path.join(input_path, 'Aids2.csv'))
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

    # save interim
    save_files('interim', 'aids_x_train.csv', X_train)
    save_files('interim', 'aids_x_test.csv', X_test)
    save_files('interim', 'aids_y_train.csv', y_train)
    save_files('interim', 'aids_y_test.csv', y_test)

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
    save_files('processed', 'aids_X_train.csv', X_train)
    save_files('processed', 'aids_X_test.csv', X_test)
    save_files('processed', 'aids_y_train.csv', y_train, header=False)
    save_files('processed', 'aids_y_test.csv', y_test, header=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # process_aids_data(input_filepath, output_filepath)
    # process_interim_weather_data(input_filepath, output_filepath)
    process_final_weather_data()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
