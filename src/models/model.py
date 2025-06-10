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
        
        # 置信区间相关配置
        self.use_uncertainty = config.get('use_uncertainty', config['model'].get('use_uncertainty', False))
        self.mc_dropout_samples = config.get('mc_dropout_samples', config['model'].get('mc_dropout_samples', 100))
        self.uncertainty_method = config.get('uncertainty_method', config['model'].get('uncertainty_method', 'variance'))  # 'variance' 或 'mc_dropout'
        
        # TODO: add target weights
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 添加额外的dropout层用于MC Dropout
        self.mc_dropout = nn.Dropout(self.dropout)
        
        if self.use_multi_task:
            self.task_heads = nn.ModuleDict()
            for target in self.prediction_targets:
                if self.use_uncertainty and self.uncertainty_method == 'variance':
                    # 为每个任务创建均值和方差输出头
                    self.task_heads[f"{target}_mean"] = nn.Sequential(
                        nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(config['model']['head_hidden_size'], self.prediction_steps)
                    )
                    self.task_heads[f"{target}_logvar"] = nn.Sequential(
                        nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(config['model']['head_hidden_size'], self.prediction_steps)
                    )
                else:
                    self.task_heads[target] = nn.Sequential(
                        nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(config['model']['head_hidden_size'], self.prediction_steps)  # 预测多个时间步
                    )
        else:
            if self.use_uncertainty and self.uncertainty_method == 'variance':
                # 单任务模式下的均值和方差输出
                self.fc_mean = nn.Sequential(
                    nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(config['model']['head_hidden_size'], self.output_size * self.prediction_steps)
                )
                self.fc_logvar = nn.Sequential(
                    nn.Linear(self.hidden_size, config['model']['head_hidden_size']),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(config['model']['head_hidden_size'], self.output_size * self.prediction_steps)
                )
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
    
    def forward(self, x, return_uncertainty=False, mc_samples=None):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            return_uncertainty: 是否返回不确定性估计
            mc_samples: Monte Carlo采样次数（仅在MC Dropout模式下使用）
        Returns:
            如果return_uncertainty=False: 返回预测值
            如果return_uncertainty=True: 返回字典包含预测值、不确定性等信息
        """
        if return_uncertainty and self.uncertainty_method == 'mc_dropout':
            return self._forward_mc_dropout(x, mc_samples or self.mc_dropout_samples)
        else:
            return self._forward_standard(x, return_uncertainty)
    
    def _forward_standard(self, x, return_uncertainty=False):
        """标准前向传播"""
        batch_size = x.size(0)
        
        # shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # 在推理时也应用MC Dropout（如果启用）
        if return_uncertainty and self.uncertainty_method == 'mc_dropout':
            last_output = self.mc_dropout(last_output)
        
        if self.use_multi_task:
            if self.use_uncertainty and self.uncertainty_method == 'variance':
                outputs = {}
                for target in self.prediction_targets:
                    mean = self.task_heads[f"{target}_mean"](last_output)
                    logvar = self.task_heads[f"{target}_logvar"](last_output)
                    
                    if return_uncertainty:
                        outputs[target] = {
                            'mean': mean,
                            'logvar': logvar,
                            'std': torch.exp(0.5 * logvar),
                            'prediction': mean  # 默认预测值为均值
                        }
                    else:
                        outputs[target] = mean
                return outputs
            else:
                outputs = {}
                for target in self.prediction_targets:
                    outputs[target] = self.task_heads[target](last_output)
                return outputs
        else:
            if self.use_uncertainty and self.uncertainty_method == 'variance':
                mean = self.fc_mean(last_output)
                logvar = self.fc_logvar(last_output)
                mean = mean.view(batch_size, self.prediction_steps)
                logvar = logvar.view(batch_size, self.prediction_steps)
                
                if return_uncertainty:
                    return {
                        'mean': mean,
                        'logvar': logvar,
                        'std': torch.exp(0.5 * logvar),
                        'prediction': mean
                    }
                else:
                    return mean
            else:
                output = self.fc(last_output)
                return output.view(batch_size, self.prediction_steps)
    
    def _forward_mc_dropout(self, x, mc_samples):
        """Monte Carlo Dropout前向传播"""
        batch_size = x.size(0)
        
        # 启用训练模式以保持dropout活跃
        self.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(mc_samples):
                # LSTM部分
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                
                # 应用MC Dropout
                last_output = self.mc_dropout(last_output)
                
                if self.use_multi_task:
                    sample_outputs = {}
                    for target in self.prediction_targets:
                        sample_outputs[target] = self.task_heads[target](last_output)
                    predictions.append(sample_outputs)
                else:
                    output = self.fc(last_output)
                    predictions.append(output.view(batch_size, self.prediction_steps))
        
        # 恢复eval模式
        self.eval()
        
        # 计算统计量
        if self.use_multi_task:
            outputs = {}
            for target in self.prediction_targets:
                target_predictions = torch.stack([pred[target] for pred in predictions], dim=0)  # [mc_samples, batch_size, prediction_steps]
                
                mean = torch.mean(target_predictions, dim=0)
                std = torch.std(target_predictions, dim=0)
                
                outputs[target] = {
                    'mean': mean,
                    'std': std,
                    'prediction': mean,
                    'samples': target_predictions
                }
            return outputs
        else:
            all_predictions = torch.stack(predictions, dim=0)  # [mc_samples, batch_size, prediction_steps]
            
            mean = torch.mean(all_predictions, dim=0)
            std = torch.std(all_predictions, dim=0)
            
            return {
                'mean': mean,
                'std': std,
                'prediction': mean,
                'samples': all_predictions
            }
    
    def predict_with_confidence(self, x, confidence_level=0.95):
        """
        带置信区间的预测
        Args:
            x: 输入张量
            confidence_level: 置信水平 (0.95 表示95%置信区间)
        Returns:
            包含预测值、置信区间上下界的字典
        """
        result = self.forward(x, return_uncertainty=True)
        
        # 计算置信区间
        z_score = torch.tensor(1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645)
        
        if self.use_multi_task:
            outputs = {}
            for target in self.prediction_targets:
                target_result = result[target]
                std = target_result['std']
                mean = target_result['mean']
                
                outputs[target] = {
                    'prediction': mean,
                    'lower_bound': mean - z_score * std,
                    'upper_bound': mean + z_score * std,
                    'std': std,
                    'confidence_level': confidence_level
                }
            return outputs
        else:
            std = result['std']
            mean = result['mean']
            
            return {
                'prediction': mean,
                'lower_bound': mean - z_score * std,
                'upper_bound': mean + z_score * std,
                'std': std,
                'confidence_level': confidence_level
            }
    
    def uncertainty_loss(self, outputs, targets):
        """
        不确定性感知的损失函数
        适用于方差预测方法
        """
        if not (self.use_uncertainty and self.uncertainty_method == 'variance'):
            raise ValueError("uncertainty_loss只能在use_uncertainty=True且uncertainty_method='variance'时使用")
        
        total_loss = 0
        
        if self.use_multi_task:
            for target_name in self.prediction_targets:
                target_idx = self.feature_columns.index(target_name)
                target_feature = targets[:, :, target_idx:target_idx+1].squeeze(-1)  # [batch_size, prediction_steps]
                
                mean = outputs[target_name]['mean']
                logvar = outputs[target_name]['logvar']
                
                # 负对数似然损失 (假设高斯分布)
                precision = torch.exp(-logvar)
                loss = 0.5 * (precision * (target_feature - mean)**2 + logvar)
                total_loss += torch.mean(loss)
        else:
            if outputs['mean'].shape != targets.shape:
                target_idx = self.feature_columns.index(self.prediction_targets[0])
                targets = targets[:, :, target_idx:target_idx+1].squeeze(-1)  # [batch_size, prediction_steps]
            
            mean = outputs['mean']
            logvar = outputs['logvar']
            
            # 负对数似然损失
            precision = torch.exp(-logvar)
            loss = 0.5 * (precision * (targets - mean)**2 + logvar)
            total_loss = torch.mean(loss)
        
        return total_loss
        