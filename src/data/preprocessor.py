"""
数据预处理模块
包含标准化、归一化和特征工程
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from pathlib import Path


class DataPreprocessor:
    """数据预处理基类"""
    
    def __init__(self, method: str = 'standard'):
        """
        初始化预处理器
        
        Args:
            method: 预处理方法 ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.is_fitted = False
        self.scalers = {}
        
    def _create_scaler(self):
        """创建缩放器"""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"不支持的预处理方法: {self.method}")
    
    def fit(self, data: torch.Tensor):
        """拟合预处理器"""
        raise NotImplementedError
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """转换数据"""
        raise NotImplementedError
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆转换数据"""
        raise NotImplementedError
    
    def save(self, path: str):
        """保存预处理器"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'is_fitted': self.is_fitted,
                'scalers': self.scalers
            }, f)
    
    def load(self, path: str):
        """加载预处理器"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.method = state['method']
            self.is_fitted = state['is_fitted']
            self.scalers = state['scalers']


class BatteryPreprocessor(DataPreprocessor):
    """电池数据预处理器"""
    
    def __init__(self, method: str = 'standard', clip_outliers: bool = True):
        super().__init__(method)
        self.clip_outliers = clip_outliers
        self.feature_stats = {}
        
    def fit(self, data: torch.Tensor):
        """
        拟合电池数据预处理器
        
        Args:
            data: 形状为 [num_samples, sequence_length, num_features] 的数据
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # 重塑数据为 [num_samples * sequence_length, num_features]
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 为每个特征创建缩放器
        self.scalers = {}
        self.feature_stats = {}
        
        for i in range(data_np.shape[1]):
            feature_data = data_np[:, i:i+1]
            
            # 移除异常值（可选）
            if self.clip_outliers:
                q1, q3 = np.percentile(feature_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                feature_data = np.clip(feature_data, lower_bound, upper_bound)
            
            # 创建并拟合缩放器
            scaler = self._create_scaler()
            scaler.fit(feature_data)
            self.scalers[i] = scaler
            
            # 保存统计信息
            self.feature_stats[i] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data))
            }
        
        self.is_fitted = True
        print(f"电池预处理器已拟合，特征数: {len(self.scalers)}")
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """转换电池数据"""
        if not self.is_fitted:
            # 如果未拟合，先用当前数据进行拟合
            print("预处理器未拟合，使用当前数据进行拟合...")
            self.fit(data)
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 转换每个特征
        transformed_data = np.zeros_like(data_np)
        
        for i in range(data_np.shape[1]):
            if i in self.scalers:
                feature_data = data_np[:, i:i+1]
                
                # 裁剪异常值（可选）
                if self.clip_outliers:
                    stats = self.feature_stats[i]
                    # 使用3倍标准差作为裁剪边界
                    lower_bound = stats['mean'] - 3 * stats['std']
                    upper_bound = stats['mean'] + 3 * stats['std']
                    feature_data = np.clip(feature_data, lower_bound, upper_bound)
                
                transformed_data[:, i:i+1] = self.scalers[i].transform(feature_data)
            else:
                transformed_data[:, i] = data_np[:, i]
        
        # 重塑回原始形状
        result = torch.tensor(transformed_data, dtype=torch.float32)
        result = result.view(original_shape)
        
        return result
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆转换电池数据"""
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合")
        
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 逆转换每个特征
        inverse_data = np.zeros_like(data_np)
        
        for i in range(data_np.shape[1]):
            if i in self.scalers:
                inverse_data[:, i:i+1] = self.scalers[i].inverse_transform(data_np[:, i:i+1])
            else:
                inverse_data[:, i] = data_np[:, i]
        
        result = torch.tensor(inverse_data, dtype=torch.float32)
        result = result.view(original_shape)
        
        return result


class FlightPreprocessor(DataPreprocessor):
    """飞行数据预处理器"""
    
    def __init__(self, method: str = 'minmax', normalize_angles: bool = True):
        super().__init__(method)
        self.normalize_angles = normalize_angles
        self.angle_indices = [3, 4, 5]  # roll, pitch, yaw的索引
        self.feature_stats = {}
        
    def fit(self, data: torch.Tensor):
        """
        拟合飞行数据预处理器
        
        Args:
            data: 形状为 [num_samples, sequence_length, num_features] 的数据
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # 重塑数据
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 为每个特征创建缩放器
        self.scalers = {}
        self.feature_stats = {}
        
        for i in range(data_np.shape[1]):
            feature_data = data_np[:, i:i+1]
            
            # 角度特征特殊处理
            if self.normalize_angles and i in self.angle_indices:
                # 将角度转换到 [-π, π] 范围
                feature_data = np.arctan2(np.sin(feature_data), np.cos(feature_data))
                
                # 对角度使用特殊的归一化到 [-1, 1]
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = self._create_scaler()
            
            scaler.fit(feature_data)
            self.scalers[i] = scaler
            
            # 保存统计信息
            self.feature_stats[i] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'is_angle': i in self.angle_indices
            }
        
        self.is_fitted = True
        print(f"飞行预处理器已拟合，特征数: {len(self.scalers)}")
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """转换飞行数据"""
        if not self.is_fitted:
            # 如果未拟合，先用当前数据进行拟合
            print("飞行预处理器未拟合，使用当前数据进行拟合...")
            self.fit(data)
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 转换每个特征
        transformed_data = np.zeros_like(data_np)
        
        for i in range(data_np.shape[1]):
            if i in self.scalers:
                feature_data = data_np[:, i:i+1]
                
                # 角度特征特殊处理
                if self.normalize_angles and i in self.angle_indices:
                    # 将角度规范化到 [-π, π]
                    feature_data = np.arctan2(np.sin(feature_data), np.cos(feature_data))
                
                transformed_data[:, i:i+1] = self.scalers[i].transform(feature_data)
            else:
                transformed_data[:, i] = data_np[:, i]
        
        # 重塑回原始形状
        result = torch.tensor(transformed_data, dtype=torch.float32)
        result = result.view(original_shape)
        
        return result
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆转换飞行数据"""
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合")
        
        original_shape = data.shape
        data_reshaped = data.view(-1, original_shape[-1])
        data_np = data_reshaped.numpy()
        
        # 逆转换每个特征
        inverse_data = np.zeros_like(data_np)
        
        for i in range(data_np.shape[1]):
            if i in self.scalers:
                inverse_data[:, i:i+1] = self.scalers[i].inverse_transform(data_np[:, i:i+1])
            else:
                inverse_data[:, i] = data_np[:, i]
        
        result = torch.tensor(inverse_data, dtype=torch.float32)
        result = result.view(original_shape)
        
        return result
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        if not self.is_fitted:
            return {}
        
        feature_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                        'velocity', 'acceleration', 'altitude']
        
        info = {}
        for i, name in enumerate(feature_names[:len(self.feature_stats)]):
            info[name] = self.feature_stats.get(i, {})
        
        return info