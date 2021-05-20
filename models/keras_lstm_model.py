from keras import Model
from keras.layers import LSTM, Input, Dense
from keras.optimizers import Adam
import tensorflow as tf
from models.base_model import BaseModel


class LSTMModel(BaseModel, tf.keras.Model):
    def __init__(self, config, seq_len, n_features, output_dim, learning_rate, loss, **kwargs):
        BaseModel.__init__(self, config)
        tf.keras.Model.__init__(self, kwargs)
        self.build_model(seq_len, n_features, output_dim, learning_rate, loss)

    def call(self, inputs):
        x = self.lstm_out(inputs)
        x = self.outputs(x)
        return x

    def build_model(self, seq_len, n_features, output_dim, learning_rate, loss):
        #lstm_out = keras.layers.LSTM(100, dropout = 0.2, recurrent_dropout = 0.2)(inputs)
        self.lstm_out = LSTM(100)
        #dense_1 = keras.layers.Dense(8)(lstm_out)
        self.outputs = Dense(output_dim)

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
