import torch
from torch.utils.data import Dataset


def get_dataset(data_path, mode: str):
    if mode == "train":
        return EVTOLDataset(data_path)
    elif mode == "val":
        return EVTOLDataset(data_path)
    elif mode == "test":
        return EVTOLDataset(data_path)

class EVTOLDataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.data_path = data_path
        self.transform = transform
        self._load_data()
        
    def _load_data(self):
        # TODO
        raise NotImplementedError("Not implemented")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]