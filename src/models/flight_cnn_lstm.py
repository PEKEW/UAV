"""
飞行异常检测CNN-LSTM混合模型
专门针对飞行数据特征优化的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

from .base_model import BaseAnomalyModel, ResidualBlock, MultiHeadAttention, AttentionPooling


class FlightFeatureExtractor(nn.Module):
    """飞行特征工程模块"""
    
    def __init__(self, input_features: int = 9):
        super().__init__()
        self.input_features = input_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        飞行特征工程
        输入: [batch_size, seq_len, 9] - [x, y, z, roll, pitch, yaw, velocity, acceleration, altitude]
        输出: [batch_size, seq_len, 16] - 原始特征 + 工程特征
        """
        batch_size, seq_len, _ = x.size()
        
        # 原始特征分解
        position = x[:, :, 0:3]        # [x, y, z]
        attitude = x[:, :, 3:6]        # [roll, pitch, yaw]
        velocity = x[:, :, 6:7]        # velocity
        acceleration = x[:, :, 7:8]    # acceleration
        altitude = x[:, :, 8:9]        # altitude
        
        # 工程特征
        engineered_features = []
        
        # 1. 角速度特征 (姿态角的时间导数)
        if seq_len > 1:
            angular_velocity = torch.diff(attitude, dim=1, prepend=attitude[:, 0:1])
        else:
            angular_velocity = torch.zeros_like(attitude)
        
        engineered_features.append(angular_velocity)
        
        # 2. 3D运动特征
        # 速度向量幅值
        if seq_len > 1:
            position_diff = torch.diff(position, dim=1, prepend=position[:, 0:1])
            velocity_3d_magnitude = torch.norm(position_diff, dim=-1, keepdim=True)
        else:
            velocity_3d_magnitude = torch.zeros_like(velocity)
        
        # 加速度向量幅值 
        acceleration_3d_magnitude = acceleration  # 已有标量加速度
        
        engineered_features.extend([velocity_3d_magnitude, acceleration_3d_magnitude])
        
        # 3. 空间几何特征
        # 距离原点的距离
        distance_from_origin = torch.norm(position, dim=-1, keepdim=True)
        
        # 高度变化率
        if seq_len > 1:
            altitude_change_rate = torch.diff(altitude, dim=1, prepend=altitude[:, 0:1])
        else:
            altitude_change_rate = torch.zeros_like(altitude)
        
        engineered_features.extend([distance_from_origin, altitude_change_rate])
        
        # 4. 飞行动力学特征
        # 俯仰角和速度的关系（爬升/下降指示器）
        pitch = attitude[:, :, 1:2]  # pitch角
        climb_indicator = pitch * velocity  # 俯仰角 * 速度
        
        # 转弯指示器（偏航角变化 * 横滚角）
        if seq_len > 1:
            yaw_rate = torch.diff(attitude[:, :, 2:3], dim=1, prepend=attitude[:, 0:1, 2:3])
        else:
            yaw_rate = torch.zeros_like(attitude[:, :, 2:3])
        
        roll = attitude[:, :, 0:1]  # roll角
        turn_indicator = yaw_rate * roll
        
        engineered_features.extend([climb_indicator, turn_indicator])
        
        # 拼接所有特征
        all_features = torch.cat([x] + engineered_features, dim=-1)
        
        return all_features


class FlightAnomalyNet(BaseAnomalyModel):
    """飞行异常检测CNN-LSTM网络"""
    
    def __init__(self, config: Dict[str, Any]):
        # 设置飞行数据默认配置
        flight_config = {
            'sequence_length': 30,
            'input_features': 9,  # 原始飞行特征数
            'num_classes': 2,
            'dropout_rate': 0.15,
            'cnn_channels': [96, 192, 384],
            'lstm_hidden': 256,
            'attention_heads': 8,
            'classifier_hidden': [128, 64]
        }
        flight_config.update(config)
        
        super().__init__(flight_config)
        
        # 特征工程模块
        self.feature_extractor = FlightFeatureExtractor(self.input_features)
        enhanced_features = 18  # 9原始 + 9工程特征
        
        # 多尺度CNN特征提取（针对飞行模式优化）
        self.cnn_branches = nn.ModuleList([
            # 分支1: 快速机动检测 (kernel=3)
            nn.Sequential(
                nn.Conv1d(enhanced_features, 96, kernel_size=3, padding=1),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ),
            # 分支2: 标准飞行模式检测 (kernel=5)
            nn.Sequential(
                nn.Conv1d(enhanced_features, 96, kernel_size=5, padding=2),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ),
            # 分支3: 扩展飞行阶段检测 (kernel=9)
            nn.Sequential(
                nn.Conv1d(enhanced_features, 96, kernel_size=9, padding=4),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(288, 192, kernel_size=1),  # 3*96 -> 192
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True)
        )
        
        # 飞行动力学特定的残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(192, 192, kernel_size=3, dropout=0.1),
            ResidualBlock(192, 384, kernel_size=3, dropout=0.15),
            ResidualBlock(384, 384, kernel_size=3, dropout=0.2),
        ])
        
        # 自适应池化（保留关键飞行阶段）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(12)
        
        # 双层双向LSTM（捕获复杂的飞行轨迹模式）
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=384,
                hidden_size=192,  # 双向后为384
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            ),
            nn.LSTM(
                input_size=384,
                hidden_size=128,  # 双向后为256
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            )
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(384),
            nn.LayerNorm(256)
        ])
        
        # Dropout层
        self.lstm_dropout = nn.Dropout(0.25)
        
        # 多头注意力机制（用于复杂飞行模式识别）
        self.multi_head_attention = MultiHeadAttention(
            d_model=256,
            num_heads=self.config['attention_heads'],
            dropout=0.2
        )
        
        # 注意力池化
        self.attention_pool = AttentionPooling(256)
        
        # 飞行特定的分类头
        classifier_layers = []
        prev_dim = 256
        
        for hidden_dim in self.config['classifier_hidden']:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(0.35 if prev_dim > 200 else 0.25)
            ])
            prev_dim = hidden_dim
            
        classifier_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        # 特征工程
        x = self.feature_extractor(x)  # [batch, seq, 16]
        
        # 转换为CNN格式 [batch, channels, seq]
        x = x.transpose(1, 2)
        
        # 多尺度CNN特征提取
        branch_outputs = []
        for branch in self.cnn_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # 特征融合
        x = torch.cat(branch_outputs, dim=1)  # [batch, 288, seq]
        x = self.feature_fusion(x)  # [batch, 192, seq]
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)  # [batch, 384, 12]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # CNN特征提取
        cnn_features = self.extract_features(x)  # [batch, 384, 12]
        
        # 转换为LSTM格式 [batch, seq, features]
        lstm_input = cnn_features.transpose(1, 2)  # [batch, 12, 384]
        
        # 第一层LSTM
        lstm_out, _ = self.lstm_layers[0](lstm_input)  # [batch, 12, 384]
        lstm_out = self.layer_norms[0](lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 第二层LSTM
        lstm_out, _ = self.lstm_layers[1](lstm_out)  # [batch, 12, 256]
        lstm_out = self.layer_norms[1](lstm_out)
        
        # 多头注意力
        lstm_out, self.multihead_attention_weights = self.multi_head_attention(lstm_out)
        
        # 注意力池化
        pooled_features, self.attention_weights = self.attention_pool(lstm_out)
        
        # 分类
        logits = self.classifier(pooled_features)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取所有注意力权重"""
        _ = self.forward(x)  # 触发前向传播计算注意力权重
        
        return {
            'temporal_attention': getattr(self, 'attention_weights', None),
            'multihead_attention': getattr(self, 'multihead_attention_weights', None)
        }
    
    def get_flight_feature_importance(self, x: torch.Tensor) -> Dict[str, Any]:
        """获取飞行特征重要性分析"""
        self.eval()
        with torch.no_grad():
            # 获取注意力权重
            attention_weights = self.get_attention_weights(x)
            
            # 获取特征激活
            enhanced_features = self.feature_extractor(x)
            cnn_features = self.extract_features(x)
            
            # 飞行特征名称
            feature_names = [
                'x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                'velocity', 'acceleration', 'altitude',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                'velocity_3d_magnitude', 'acceleration_3d_magnitude',
                'distance_from_origin', 'altitude_change_rate',
                'climb_indicator', 'turn_indicator'
            ]
            
            # 计算每个特征的平均激活
            feature_activations = enhanced_features.abs().mean(dim=(0, 1))
            
            # 飞行阶段分析
            flight_phases = self._analyze_flight_phases(x)
            
            importance_dict = {
                'attention_weights': attention_weights,
                'feature_activations': dict(zip(feature_names, feature_activations)),
                'flight_phases': flight_phases,
                'temporal_importance': attention_weights['temporal_attention'].mean(dim=0) 
                                     if attention_weights['temporal_attention'] is not None else None
            }
            
        return importance_dict
    
    def _analyze_flight_phases(self, x: torch.Tensor) -> Dict[str, Any]:
        """分析飞行阶段"""
        # 提取关键飞行参数
        altitude = x[:, :, 8]  # altitude
        velocity = x[:, :, 6]  # velocity
        
        # 简单的飞行阶段识别
        altitude_change = torch.diff(altitude, dim=1, prepend=altitude[:, 0:1])
        
        phases = {
            'takeoff': (altitude_change > 0.1).float().mean(),
            'cruise': ((altitude_change.abs() < 0.05) & (velocity > 5)).float().mean(),
            'landing': (altitude_change < -0.1).float().mean(),
            'hover': ((velocity < 2) & (altitude_change.abs() < 0.02)).float().mean()
        }
        
        return phases