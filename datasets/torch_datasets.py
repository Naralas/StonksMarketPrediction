from torch.utils.data import Dataset
import pandas as pd
import torch

class StocksDataset(Dataset):
    """Pytorch dataset class for stocks.

    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset object
    """
    def __init__(self, data, target):
        """Init function of the dataset. Will transform data and target to Pytorch Tensors.

        Args:
            data (Numpy Array): Numpy array of samples features (X).
            target (Numpy Array): Numpy array of targets (y).
        """
        self.data = torch.Tensor(data)
        self.target = torch.Tensor(target)
        
    def __getitem__(self, index):
        """Get item at specific position (array access).

        Args:
            index (int): Index.

        Returns:
            Tuple(Pytorch tensor, Pytorch tensor): Tuple of pytorch tensors of sample features and target.
        """
        datapoint = self.data[index]
        target = self.target[index]
        return datapoint, target
        
    def __len__(self):
        """Returns the length of the dataset 

        Returns:
            Int: Length of the dataset.
        """
        return len(self.data)
    

