"""
简单电池异常分类CNN模型
用于快速测试数据集是否合理的轻量级模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

from .base_model import BaseAnomalyModel


class SimpleBatteryCNN(BaseAnomalyModel):
    """简单电池异常分类CNN模型"""
    
    def __init__(self, config: Dict[str, Any]):
        # 设置默认配置
        default_config = {
            'sequence_length': 30,
            'input_features': 7,  # 电池原始特征数
            'num_classes': 2,
            'cnn_channels': [32, 64, 128],  # 简化的通道数
            'dropout_rate': 0.5,
            'use_batch_norm': True
        }
        default_config.update(config)
        
        super().__init__(default_config)
        
        # 特征归一化层
        self.feature_norm = nn.LayerNorm(self.input_features)
        
        # 简单的CNN特征提取器
        self.cnn_layers = nn.ModuleList()
        in_channels = self.input_features
        
        for i, out_channels in enumerate(self.config['cnn_channels']):
            # 卷积块
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels) if self.config['use_batch_norm'] else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config['dropout_rate']),
                nn.MaxPool1d(kernel_size=2, stride=2)  # 逐步减少序列长度
            )
            self.cnn_layers.append(conv_block)
            in_channels = out_channels
        
        # 计算经过CNN后的序列长度
        self.cnn_output_length = self.sequence_length
        for _ in self.config['cnn_channels']:
            self.cnn_output_length = self.cnn_output_length // 2
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        classifier_input_dim = self.config['cnn_channels'][-1]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config['dropout_rate']),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config['dropout_rate']),
            nn.Linear(32, self.num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        # 输入: [batch_size, seq_len, features]
        batch_size = x.size(0)
        
        # 特征归一化
        x = self.feature_norm(x)
        
        # 转换为CNN输入格式: [batch_size, features, seq_len]
        x = x.transpose(1, 2)
        
        # CNN特征提取
        for conv_layer in self.cnn_layers:
            x = conv_layer(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征提取
        features = self.extract_features(x)
        
        # 全局平均池化
        pooled = self.global_pool(features).squeeze(-1)  # [batch_size, channels]
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取中间特征图，用于可视化"""
        feature_maps = {}
        
        # 特征归一化
        x = self.feature_norm(x)
        x = x.transpose(1, 2)
        
        # 记录每一层的输出
        for i, conv_layer in enumerate(self.cnn_layers):
            x = conv_layer(x)
            feature_maps[f'conv_{i+1}'] = x.clone()
        
        return feature_maps
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SimpleBatteryCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': (self.sequence_length, self.input_features),
            'output_shape': (self.num_classes,),
            'cnn_channels': self.config['cnn_channels'],
            'cnn_output_length': self.cnn_output_length,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }


class SimpleBatteryCNNWithAttention(SimpleBatteryCNN):
    """带注意力机制的简单电池CNN模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 添加简单的注意力机制
        attention_dim = self.config['cnn_channels'][-1]
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.Tanh(),
            nn.Linear(attention_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的前向传播"""
        # 特征提取
        features = self.extract_features(x)  # [batch_size, channels, seq_len]
        
        # 转换为注意力输入格式: [batch_size, seq_len, channels]
        features_t = features.transpose(1, 2)
        
        # 计算注意力权重
        attention_scores = self.attention(features_t)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # 加权聚合
        attended_features = torch.sum(features_t * attention_weights, dim=1)  # [batch_size, channels]
        
        # 分类
        logits = self.classifier(attended_features)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重"""
        features = self.extract_features(x)
        features_t = features.transpose(1, 2)
        attention_scores = self.attention(features_t)
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights.squeeze(-1)  # [batch_size, seq_len] 