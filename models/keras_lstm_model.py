from keras import Model
from keras.layers import LSTM, Input, Dense
from keras.optimizers import Adam
import tensorflow as tf
from models.base_model import BaseModel


class LSTMModel(BaseModel, tf.keras.Model):
    """
    Args:
        BaseModel: Inherited base model classes that contains the config, etc.
        tf.Keras.Model: Inherited base keras model class so this object can be used as a "regular" Keras model

    """
    def __init__(self, config, seq_len, n_features, output_dim, learning_rate, loss, **kwargs):
        """Init function for the model. Will build the model using build_model(...)

        Args:
            config (dict): Dictionary config that will used for setup and be stored in wandb
            seq_len (int): Number of stock periods to include
            n_features (int): Input dimension 
            output_dim (int): Output dimension
            learning_rate (float): Learning rate
            loss (tf.keras.losses Loss function): Keras loss function 
        """
        BaseModel.__init__(self, config)
        tf.keras.Model.__init__(self, kwargs)
        self.build_model(seq_len, n_features, output_dim, learning_rate, loss)

    def call(self, inputs):
        """Overriden call function of the keras model. Make a pass of the inputs through the model and outputs the results.

        Args:
            inputs (Keras dataset or Numpy array): Input samples

        Returns:
            float array: Output array or single sample, depending on the output_dim given
        """
        x = self.lstm_out(inputs)
        x = self.outputs(x)
        return x

    def build_model(self, seq_len, n_features, output_dim, learning_rate, loss):
        """Builds the keras model. Uses adam optimizer with given learning rate.

        Args:
            seq_len (int): Number of stock periods to include
            n_features (int): Input dimension 
            output_dim (int): Output dimension
            learning_rate (float): Learning rate
            loss (tf.keras.losses Loss function): Keras loss function 
        """
        self.lstm_out = LSTM(100)
        self.outputs = Dense(output_dim)

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
