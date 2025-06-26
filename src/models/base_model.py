"""
异常检测模型基类
定义了模型的通用接口和功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseAnomalyModel(nn.Module, ABC):
    """异常检测模型抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.sequence_length = config.get('sequence_length', 30)
        self.num_classes = config.get('num_classes', 2)
        self.input_features = config.get('input_features', 7)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取层，由子类实现"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，由子类实现"""
        pass
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重，用于可解释性分析"""
        return None
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率分布"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """二分类预测"""
        proba = self.predict_proba(x)
        return (proba[:, 1] > threshold).long()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': (self.sequence_length, self.input_features),
            'output_shape': (self.num_classes,),
            'config': self.config
        }


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        
        # Self-attention
        residual = x
        x = self.layer_norm(x)
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影和残差连接
        output = self.w_o(context)
        output = output + residual
        
        # 返回平均注意力权重用于可视化
        avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        return output, avg_attention


class ResidualBlock(nn.Module):
    """残差卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接的维度匹配
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        out += residual
        out = self.relu(out)
        
        return out


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, hidden_dim]
        attention_scores = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # 加权聚合
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return pooled, attention_weights.squeeze(-1)  # 返回池化结果和注意力权重