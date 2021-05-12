from models.base_model import BaseModel
from keras import Model
from keras.layers import LSTM, Input, Dense
from keras.optimizers import Adam

class LSTMModel(BaseModel):
    def __init__(self, config, inputs, learning_rate, loss):
        super(LSTMModel, self).__init__(config)
        self.build_model(inputs, learning_rate, loss)

    def build_model(self, inputs, learning_rate, loss):
        inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
        #lstm_out = keras.layers.LSTM(100, dropout = 0.2, recurrent_dropout = 0.2)(inputs)
        lstm_out = LSTM(100)(inputs)
        #dense_1 = keras.layers.Dense(8)(lstm_out)
        outputs = Dense(1)(lstm_out)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
