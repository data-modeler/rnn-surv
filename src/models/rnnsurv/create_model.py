'''
    Create Model
    -----

    Creates the RNN-SURV model based on hyperparameters.
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout, Reshape, Flatten, Lambda

from .losses import loss1, loss2


def create_model(n_features, 
                 dense_sizes = (20, 10),
                 lstm_sizes = (30, 30),
                 dropout_prob = 0.5,
                 max_length = 300,
                 pad_token = -999,
                 optimizer = 'adam',
                 loss_weights = {"y_hat": 1.0, "r_out": 1.0}
                 ):
    '''Creates the RNN-SURV based on hyperparameters.
    '''
    # define the model architecture
    x_in = Input(batch_shape=(None, max_length, n_features), name='x_in')
    loss_mask = Input(batch_shape=(None, max_length), name='loss_mask')
    flip_i = Input(batch_shape=(None, None), name='flip_i')

    dense_layers = Sequential(name='denses')
    for i, d_size in enumerate(dense_sizes):
        if i == 0:
            dense_layers.add(
                TimeDistributed(
                    Dense(d_size, activation='relu'),
                          input_shape=(max_length, n_features)))
            dense_layers.add(Dropout(dropout_prob))
        else:
            dense_layers.add(
                TimeDistributed(
                    Dense(d_size, activation='relu'),
                          input_shape=(max_length, dense_sizes[i-1])))
            dense_layers.add(Dropout(dropout_prob))

    layers = Sequential(name='y_hat')
    for l_size in lstm_sizes:
        layers.add(LSTM(l_size, return_sequences=True))

    layers.add(Dropout(dropout_prob))
    layers.add(TimeDistributed(Dense(1, activation='sigmoid')))
    layers.add(Flatten())

    calculate_r = Dense(1, activation='linear', name='r_out')

    # connect the network layers
    denses = dense_layers(x_in)
    y_hat = layers(denses)
    r_out = calculate_r(y_hat)

    losses = {
            "y_hat": loss1(pad_token),
            "r_out": loss2(loss_mask, pad_token, flip_i),
    }

    model = Model([x_in, loss_mask, flip_i], [r_out, y_hat])
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights,
                  experimental_run_tf_function=False)
    # the argument `experimental_run_tf_function=False` was added per this issue:
    # https://github.com/tensorflow/probability/issues/519
    # this was throwing the same error
    model.summary()
    
    return model

if __name__ == '__main__':

    from data_generator import DataGenerator
    from utils import get_data

    X_TRN, Y_TRN = get_data(nrows=2000)
    N_FEATURES = X_TRN.shape[1] - 1 # all columns but `oid` are features

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

    TRAIN_GENERATOR = DataGenerator(X_TRN, Y_TRN, **PARAMS)

    MODEL.fit(TRAIN_GENERATOR, epochs=5)


