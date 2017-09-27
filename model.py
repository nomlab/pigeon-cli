import pandas as pd
import numpy as np
from datetime import timedelta
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D

class Model:
    @classmethod
    def create_from_desc(cls, input_shape):
        in_out_neurons = 3
        nfilters = 128
        hidden_neurons = 256

        model = Sequential()
        model.add(TimeDistributed(Convolution1D(nfilters,
                                        4,
                                        activation = 'relu',
                                        padding = 'same'),
                                        input_shape=input_shape))
        model.add(TimeDistributed(Convolution1D(nfilters,
                                        2,
                                        activation = 'relu',
                                        padding = 'same')))
        model.add(TimeDistributed(MaxPooling1D()))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Flatten()))

        model.add(TimeDistributed(Dense(hidden_neurons, activation = 'linear')))
        model.add(LSTM(hidden_neurons, return_sequences = True))

        model.add(TimeDistributed(Dense(hidden_neurons, activation = 'relu')))
        model.add(TimeDistributed(Dense(hidden_neurons, activation = 'relu')))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(in_out_neurons, activation = 'linear')))
        model.add(GlobalAveragePooling1D())
        model.compile(loss="mean_squared_error", optimizer="adam")

        return cls(model)

    @classmethod
    def load(cls, filepath):
        return cls(load_model(filepath))

    def __init__(self, model):
        self.model = model

    def train(self, x, y, batch_size = 100, epochs = 100, validation_split = 0.05):
        self.model.fit(x, y, batch_size = batch_size,
                       epochs = epochs, validation_split = validation_split)

    def save(self, filename):
        self.model.save(filename)

    def predict_sequences(self, x, first_dt, n_pred, data_shape):
        test_list = np.array([x])
        pred_list = np.empty((0, 3), float)
        dt_list = []
        next_dt = first_dt

        for i in range(n_pred):
            predicted = self.model.predict(test_list)
            tes = test_list[0]
            tes.shape = (data_shape[0] * data_shape[1], data_shape[2])
            tes = np.roll(tes, -1, axis = 0)
            tes[-1] = predicted
            tes.shape = data_shape
            test_list[0] = tes
            pred_list = np.append(pred_list, predicted, axis = 0)
            dt_list.append(next_dt)
            next_dt = next_dt + timedelta(hours = 1)

        dics = {}
        dics['timestamp'] = dt_list
        dics['power_usage'] = pred_list[:,0].tolist()
        dics['gas_usage'] = pred_list[:,1].tolist()
        dics['water_usage'] = pred_list[:,2].tolist()

        return dics
