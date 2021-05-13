"""

Some code inspired by https://github.com/Ahmkel/Keras-Project-Template/blob/master/base/base_model.py

"""

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None


    def build_model(self):
        raise NotImplementedError

