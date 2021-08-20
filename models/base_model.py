"""

Base model class inspired by https://github.com/Ahmkel/Keras-Project-Template/blob/master/base/base_model.py

"""

class BaseModel:
    """Base model class for Pytorch and Keras models. Stores the config and model depending on framework implementation.
    This class is an abstraction for both frameworks as the focus of which framework changed during the project.
    This class should not be used directly but rather inherited.
    """
    def __init__(self, config):
        """Init function. Stores the model

        Args:
            config (dict): Model configuration dict, will be used and stored on wandb if set to use.
        """
        self.config = config
        self.model = None

    def build_model(self):
        raise NotImplementedError()