"""
电池异常检测CNN-LSTM混合模型
专门针对电池数据特征优化的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

from .base_model import BaseAnomalyModel, ResidualBlock, AttentionPooling


class BatteryFeatureExtractor(nn.Module):
    """电池特征工程模块"""
    
    def __init__(self, input_features: int = 7):
        super().__init__()
        self.input_features = input_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Test : 不用进行特征工程 目前的网络性能足够
        return x
        """
        电池特征工程
        输入: [batch_size, seq_len, 7] - [Ecell_V, I_mA, EnergyCharge_W_h, QCharge_mA_h, 
                                        EnergyDischarge_W_h, QDischarge_mA_h, Temperature__C]
        输出: [batch_size, seq_len, 14] - 原始特征 + 工程特征
        """
        batch_size, seq_len, _ = x.size()
        
        # 原始特征
        voltage = x[:, :, 0:1]          # Ecell_V
        current = x[:, :, 1:2]          # I_mA
        energy_charge = x[:, :, 2:3]    # EnergyCharge_W_h
        charge_capacity = x[:, :, 3:4]  # QCharge_mA_h
        energy_discharge = x[:, :, 4:5] # EnergyDischarge_W_h
        discharge_capacity = x[:, :, 5:6] # QDischarge_mA_h
        temperature = x[:, :, 6:7]      # Temperature__C
        
        # 工程特征
        engineered_features = []
        
        # 1. 时间导数特征 (dV/dt, dI/dt, dT/dt) - 改进数值稳定性
        if seq_len > 1:
            dv_dt = torch.diff(voltage, dim=1, prepend=voltage[:, 0:1])
            di_dt = torch.diff(current, dim=1, prepend=current[:, 0:1]) 
            dt_dt = torch.diff(temperature, dim=1, prepend=temperature[:, 0:1])
            
            # 适度限制导数范围，保留更多信息
            dv_dt = torch.clamp(dv_dt, min=-20.0, max=20.0)
            di_dt = torch.clamp(di_dt, min=-20000.0, max=20000.0)  # 电流变化可能较大
            dt_dt = torch.clamp(dt_dt, min=-100.0, max=100.0)
        else:
            dv_dt = torch.zeros_like(voltage)
            di_dt = torch.zeros_like(current)
            dt_dt = torch.zeros_like(temperature)
            
        engineered_features.extend([dv_dt, di_dt, dt_dt])
        
        # 2. 能量效率特征 (平衡数值稳定性和信息保留)
        # 使用更安全的除法，但放宽限制以保留更多信息
        eps = 1e-3  # 保护值
        max_ratio = 50.0  # 放宽最大比值限制
        
        # 充放电效率，使用torch.clamp限制范围
        charge_efficiency = torch.clamp(
            energy_charge / torch.clamp(energy_discharge, min=eps), 
            min=0.0, max=max_ratio
        )
        
        # 容量比，同样限制范围
        capacity_ratio = torch.clamp(
            charge_capacity / torch.clamp(discharge_capacity, min=eps),
            min=0.0, max=max_ratio
        )
        
        engineered_features.extend([charge_efficiency, capacity_ratio])
        
        # 3. 功率特征 (添加数值稳定性)
        power = voltage * current / 1000.0  # 功率 (W)
        # 适度限制功率范围，保留更多信息
        power = torch.clamp(power, min=-500.0, max=500.0)
        engineered_features.append(power)
        
        # 4. 滑动统计特征 (5点窗口)
        window_size = min(5, seq_len)
        if seq_len >= window_size:
            # 对电压进行移动平均
            voltage_padded = F.pad(voltage.transpose(1, 2), 
                                 (window_size//2, window_size//2), mode='replicate')
            voltage_ma = F.avg_pool1d(voltage_padded, window_size, stride=1).transpose(1, 2)
        else:
            voltage_ma = voltage
        
        engineered_features.append(voltage_ma)
        
        # 拼接所有特征
        all_features = torch.cat([x] + engineered_features, dim=-1)
        
        # 添加特征归一化层，确保数值稳定性
        # 检查是否有NaN或Inf
        if torch.isnan(all_features).any() or torch.isinf(all_features).any():
            # 如果有异常值，用原始特征替代
            print("Warning: NaN or Inf detected in engineered features, using original features only")
            return x
        
        return all_features


class BatteryAnomalyNet(BaseAnomalyModel):
    def __init__(self, config: Dict[str, Any]):
        battery_config = {
            'sequence_length': 30,
            'input_features': 7,
            'num_classes': 2,
            'dropout_rate': 0.5,
            'cnn_channels': [64, 128, 128, 256],
            'lstm_hidden': 256,
            'attention_heads': 8,
            'classifier_hidden': [128, 64]
        }
        battery_config.update(config)
        
        super().__init__(battery_config)
        
        # Test : 不用进行特征工程 目前的网络性能足够
        self.feature_extractor = BatteryFeatureExtractor(self.input_features)
        # enhanced_features = 14
        enhanced_features = 7
        
        
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(enhanced_features, 64, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config['dropout_rate'] * 0.75)
            ) for k in [3, 7, 15]
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128, 128, kernel_size=3, dropout=self.config['dropout_rate']),
            ResidualBlock(128, 256, kernel_size=3, dropout=self.config['dropout_rate'] * 1.2),
            ResidualBlock(256, 256, kernel_size=3, dropout=self.config['dropout_rate'] * 1.5),
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(15) 
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.config['lstm_hidden'] // 2,  # 双向
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.config['dropout_rate']
        )
        
        self.layer_norm = nn.LayerNorm(self.config['lstm_hidden'])
        
        self.attention_pool = AttentionPooling(self.config['lstm_hidden'])
        
        classifier_layers = []
        prev_dim = self.config['lstm_hidden']
        
        for hidden_dim in self.config['classifier_hidden']:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(self.config['dropout_rate'])
            ])
            prev_dim = hidden_dim
            
        classifier_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        if 'bias_hh' in name:
                            hidden_size = param.size(0) // 4
                            param.data[hidden_size:2*hidden_size].fill_(2.0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)  # [batch, seq, 14]
        x = x.transpose(1, 2)
        
        branch_outputs = []
        for branch in self.cnn_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        x = torch.cat(branch_outputs, dim=1)  # [batch, 192, seq]
        x = self.feature_fusion(x)  # [batch, 128, seq]
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.adaptive_pool(x)  # [batch, 256, 15]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.extract_features(x)  # [batch, 256, 15]
        
        lstm_input = cnn_features.transpose(1, 2)  # [batch, 15, 256]
        
        lstm_out, _ = self.lstm(lstm_input)  # [batch, 15, 256]
        lstm_out = self.layer_norm(lstm_out)
        
        pooled_features, self.attention_weights = self.attention_pool(lstm_out)
        
        logits = self.classifier(pooled_features)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.forward(x)
        return getattr(self, 'attention_weights', None)
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            attention_weights = self.get_attention_weights(x)
            
            enhanced_features = self.feature_extractor(x)
            cnn_features = self.extract_features(x)
            
            feature_names = [
                'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C',
                'dV/dt', 'dI/dt', 'dT/dt', 'Charge_Efficiency', 
                'Capacity_Ratio', 'Power', 'Voltage_MA'
            ]
            
            feature_activations = enhanced_features.abs().mean(dim=(0, 1))
            
            importance_dict = {
                'attention_weights': attention_weights,
                'feature_activations': dict(zip(feature_names, feature_activations)),
                'temporal_importance': attention_weights.mean(dim=0) if attention_weights is not None else None
            }
            
        return importance_dict