import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

class Trainer:
    def __init__(self, model: nn.Model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self._setup_training()
        
    def _setup_training(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.config['train']['learning_rate'],
            weight_decay = self.config['train']['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(self.config['train']['device'])
        
    def train_epoch(self, dataloader: DataLoader):
        raise NotImplementedError("Not implemented")
    
    def validate(self, dataloader: DataLoader):
        raise NotImplementedError("Not implemented")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        raise NotImplementedError("Not implemented")