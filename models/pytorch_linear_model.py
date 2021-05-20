import torch.nn as nn
from models.base_model import BaseModel

class LinearModel(nn.Module, BaseModel):
    def __init__(self, config, input_dim, output_dim, device):
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)

        self.build_model(input_dim, output_dim, device)
        self.device = device
        self.optimizer = config['optimizer'](self.model.parameters(), lr=config['lr'])
        self.loss_fn = config['loss']()

    def forward(self, x):
        out = self.model(x)
        return out

    def build_model(self, input_dim, output_dim, device):
        model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, output_dim),
            #nn.Softmax(1),
        )

        self.model = model.to(device)
        