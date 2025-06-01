import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self._build_model(config)
        
    def _build_model(self, config):
        raise NotImplementedError("Not implemented")
    
    def forward(self, x):
        raise NotImplementedError("Not implemented")