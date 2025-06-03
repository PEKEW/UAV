import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self._build_model(config)
        
    def _build_model(self, config):
        # 输入特征不包括Ns Ns是电池串联数量，通常是固定的
        self.input_size = 9
        self.hidden_size = config.get('hidden_size', config['model']['hidden_size'])
        self.num_layers = config.get('num_layers', config['model']['num_layers'])
        self.prediction_targets = config.get('prediction_targets', config['model']['prediction_targets'])
        self.feature_columns = config.get('feature_columns', config['data']['feature_columns'])
        self.output_size = len(self.prediction_targets)
        self.dropout = config.get('dropout', config['model']['dropout'])
        self.use_multi_task = config.get('use_multi_task', config['model']['use_multi_task'])
        self.prediction_steps = config.get('prediction_steps', config['model']['prediction_steps'])
        self.padding_value = config.get('padding_value', config['model']['padding_value'])
        # TODO: add target weights
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        if self.use_multi_task:
            self.task_heads = nn.ModuleDict({
                target: nn.Sequential(
                    nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(config['model']['head_hidden_size'], self.prediction_steps)  # 预测多个时间步
                ) for target in self.prediction_targets
            })
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(config['model']['head_hidden_size'], self.output_size * self.prediction_steps)
            )
            

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        if self.use_multi_task:
            outputs = {}
            for target in self.prediction_targets:
                # 确保每个任务的输出维度为 [batch_size, prediction_steps]
                outputs[target] = self.task_heads[target](last_output)
            return outputs
        else:
            output = self.fc(last_output)
            # 重塑输出维度为 [batch_size, prediction_steps]
            return output.view(batch_size, self.prediction_steps)
        