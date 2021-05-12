from torch.utils.data import Dataset
import pandas as pd
import torch

class StocksDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.Tensor(data.values)
        self.target = torch.Tensor(target.values)
        
    def __getitem__(self, index):
        datapoint = self.data[index]
        target = self.target[index]
        return datapoint, target
        
    def __len__(self):
        return len(self.data)
    
class StocksSeqDataset(StocksDataset):
    def __init__(self, data, target):
        self.data = torch.Tensor(data)
        self.target = torch.Tensor(target.values)
    
    def __getitem__(self, index):
        datapoint = self.data[index]
        target = self.target[index]
        return datapoint, target


if __name__ == '__main__':
    test_df = pd.DataFrame({'a':[1,2,3,4,5,6,7,8,9,10], 'b':[0,0,0,0,0,1,1,1,1,1]})
    dataset = StocksSeqDataset(test_df['a'], test_df['b'], 3)
    for seq in dataset:
        print(seq)
