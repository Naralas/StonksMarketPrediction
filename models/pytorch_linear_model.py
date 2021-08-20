import torch.nn as nn
from models.base_model import BaseModel

class LinearModel(nn.Module, BaseModel):
    """This class is a Pytorch implementation for the ML models. It is a MLP model.

    Args:
        nn (nn.Module): Inherited base Pytorch model
        BaseModel (BaseModel object): Inherited base project model.
    Attributes:
        model (nn.Sequential): definitive Pytorch model object
        device (string): Device 'cuda' for gpu or 'cpu'
        optimizer (torch.optim.Optimizer object): Pytorch Optimizer object 
        loss_fn (torch.nn.Loss object): Loss function Pytorch object
    """
    def __init__(self, config, input_dim, output_dim, device):
        """Init function of the model. Will store the config, create the optimizer and build the model

        Args:
            config (dict): Dictionary config that will used for setup and be stored in wandb
            input_dim (int): Input dimension of the model (number of features)
            output_dim (int): Output dimension
            device (string): 'cpu' or 'cuda' for GPU
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)

        self.build_model(input_dim, output_dim, device)
        self.device = device
        self.config['output_dim'] = output_dim
        self.optimizer = config['optimizer'](self.model.parameters(), lr=config['lr'])
        self.loss_fn = config['loss']()

    def forward(self, x):
        """Make a forward pass through the model

        Args:
            x (nn.Tensor): Pytorch tensor of input sample(s)

        Returns:
            nn.Tensor: Output tensor of the model, size depends on batch size 
        """
        out = self.model(x)
        return out

    def build_model(self, input_dim, output_dim, device):
        """Builds the Pytorch MLP model.

        Args:
            input_dim (int): Input dimension of the model (number of features)
            output_dim (int): Output dimension
            device (string): 'cpu' or 'cuda' for GPU
        """
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
        