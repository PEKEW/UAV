"""
数据增强模块
为电池和飞行数据提供领域特定的增强方法
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import random


class DataAugmentation:
    """数据增强基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        if not self.enabled:
            return data
        return self.apply_augmentation(data)
    
    def apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """应用增强，由子类实现"""
        raise NotImplementedError


class BatteryAugmentation(DataAugmentation):
    """电池数据增强"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 增强参数
        self.time_jitter_prob = self.config.get('time_jitter_prob', 0.3)
        self.time_jitter_std = self.config.get('time_jitter_std', 0.1)
        
        self.gaussian_noise_prob = self.config.get('gaussian_noise_prob', 0.5)
        self.noise_std_ratio = self.config.get('noise_std_ratio', 0.01)
        
        self.dropout_prob = self.config.get('dropout_prob', 0.2)
        self.dropout_ratio = self.config.get('dropout_ratio', 0.05)
        
        self.voltage_fluctuation_prob = self.config.get('voltage_fluctuation_prob', 0.3)
        self.voltage_fluctuation_std = self.config.get('voltage_fluctuation_std', 0.02)
        
        self.current_spike_prob = self.config.get('current_spike_prob', 0.2)
        self.current_spike_magnitude = self.config.get('current_spike_magnitude', 0.1)
        
        self.temperature_drift_prob = self.config.get('temperature_drift_prob', 0.25)
        self.temperature_drift_range = self.config.get('temperature_drift_range', 2.0)
    
    def apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用电池数据增强
        
        Args:
            data: 形状为 [sequence_length, num_features] 的数据
        """
        augmented_data = data.clone()
        
        # 1. 时间抖动
        if random.random() < self.time_jitter_prob:
            augmented_data = self._apply_time_jitter(augmented_data)
        
        # 2. 高斯噪声
        if random.random() < self.gaussian_noise_prob:
            augmented_data = self._apply_gaussian_noise(augmented_data)
        
        # 3. 随机Dropout
        if random.random() < self.dropout_prob:
            augmented_data = self._apply_dropout(augmented_data)
        
        # 4. 电压波动（针对Ecell_V特征）
        if random.random() < self.voltage_fluctuation_prob:
            augmented_data = self._apply_voltage_fluctuation(augmented_data)
        
        # 5. 电流尖峰（针对I_mA特征）
        if random.random() < self.current_spike_prob:
            augmented_data = self._apply_current_spike(augmented_data)
        
        # 6. 温度漂移（针对Temperature__C特征）
        if random.random() < self.temperature_drift_prob:
            augmented_data = self._apply_temperature_drift(augmented_data)
        
        return augmented_data
    
    def _apply_time_jitter(self, data: torch.Tensor) -> torch.Tensor:
        """应用时间抖动"""
        seq_len, num_features = data.shape
        
        # 生成时间偏移 - 使用固定std值，可以使用size参数
        jitter = torch.normal(mean=0.0, std=self.time_jitter_std, size=(seq_len,))
        jitter = torch.cumsum(jitter, dim=0)
        
        # 限制偏移范围
        jitter = torch.clamp(jitter, -seq_len * 0.1, seq_len * 0.1)
        
        # 应用插值
        indices = torch.arange(seq_len, dtype=torch.float32) + jitter
        indices = torch.clamp(indices, 0, seq_len - 1)
        
        # 线性插值
        augmented_data = torch.zeros_like(data)
        for i in range(num_features):
            augmented_data[:, i] = torch.nn.functional.interpolate(
                data[:, i].unsqueeze(0).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return augmented_data
    
    def _apply_gaussian_noise(self, data: torch.Tensor) -> torch.Tensor:
        """应用高斯噪声"""
        # 计算每个特征的标准差
        feature_stds = torch.std(data, dim=0, keepdim=True)
        
        # 生成噪声 - 当std是Tensor时，不能使用size参数
        # 需要广播std到正确的形状
        std_tensor = self.noise_std_ratio * feature_stds.expand_as(data)
        noise = torch.normal(mean=0.0, std=std_tensor)
        
        return data + noise
    
    def _apply_dropout(self, data: torch.Tensor) -> torch.Tensor:
        """应用随机dropout"""
        seq_len = data.shape[0]
        num_dropout = int(seq_len * self.dropout_ratio)
        
        if num_dropout > 0:
            # 随机选择要dropout的时间步
            dropout_indices = random.sample(range(seq_len), num_dropout)
            augmented_data = data.clone()
            augmented_data[dropout_indices] = 0
            return augmented_data
        
        return data
    
    def _apply_voltage_fluctuation(self, data: torch.Tensor) -> torch.Tensor:
        """应用电压波动（特征索引0为Ecell_V）"""
        augmented_data = data.clone()
        
        if data.shape[1] > 0:  # 确保有Ecell_V特征
            # 使用固定std值，可以使用size参数
            voltage_noise = torch.normal(mean=0.0, std=self.voltage_fluctuation_std, size=(data.shape[0], 1))
            augmented_data[:, 0:1] += voltage_noise
        
        return augmented_data
    
    def _apply_current_spike(self, data: torch.Tensor) -> torch.Tensor:
        """应用电流尖峰（特征索引1为I_mA）"""
        augmented_data = data.clone()
        
        if data.shape[1] > 1:  # 确保有I_mA特征
            seq_len = data.shape[0]
            # 随机选择1-3个时间点添加电流尖峰
            num_spikes = random.randint(1, 3)
            spike_indices = random.sample(range(seq_len), min(num_spikes, seq_len))
            
            for idx in spike_indices:
                spike_magnitude = random.uniform(-self.current_spike_magnitude, 
                                               self.current_spike_magnitude)
                current_std = torch.std(data[:, 1])
                augmented_data[idx, 1] += spike_magnitude * current_std
        
        return augmented_data
    
    def _apply_temperature_drift(self, data: torch.Tensor) -> torch.Tensor:
        """应用温度漂移（特征索引6为Temperature__C）"""
        augmented_data = data.clone()
        
        if data.shape[1] > 6:  # 确保有Temperature__C特征
            # 生成缓慢的温度漂移
            drift = random.uniform(-self.temperature_drift_range, self.temperature_drift_range)
            drift_pattern = torch.linspace(0, drift, data.shape[0]).unsqueeze(1)
            augmented_data[:, 6:7] += drift_pattern
        
        return augmented_data


class FlightAugmentation(DataAugmentation):
    """飞行数据增强"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 增强参数
        self.trajectory_warp_prob = self.config.get('trajectory_warp_prob', 0.3)
        self.warp_magnitude = self.config.get('warp_magnitude', 0.05)
        
        self.attitude_noise_prob = self.config.get('attitude_noise_prob', 0.4)
        self.attitude_noise_std = self.config.get('attitude_noise_std', 0.1)
        
        self.velocity_perturb_prob = self.config.get('velocity_perturb_prob', 0.3)
        self.velocity_perturb_ratio = self.config.get('velocity_perturb_ratio', 0.02)
        
        self.altitude_shift_prob = self.config.get('altitude_shift_prob', 0.25)
        self.altitude_shift_range = self.config.get('altitude_shift_range', 1.0)
        
        self.gaussian_noise_prob = self.config.get('gaussian_noise_prob', 0.4)
        self.noise_std_ratio = self.config.get('noise_std_ratio', 0.01)
        
        self.time_scaling_prob = self.config.get('time_scaling_prob', 0.2)
        self.time_scaling_range = self.config.get('time_scaling_range', (0.9, 1.1))
    
    def apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用飞行数据增强
        
        Args:
            data: 形状为 [sequence_length, num_features] 的数据
                  特征顺序: [x, y, z, roll, pitch, yaw, velocity, acceleration, altitude]
        """
        augmented_data = data.clone()
        
        # 1. 轨迹扭曲
        if random.random() < self.trajectory_warp_prob:
            augmented_data = self._apply_trajectory_warp(augmented_data)
        
        # 2. 姿态噪声
        if random.random() < self.attitude_noise_prob:
            augmented_data = self._apply_attitude_noise(augmented_data)
        
        # 3. 速度扰动
        if random.random() < self.velocity_perturb_prob:
            augmented_data = self._apply_velocity_perturbation(augmented_data)
        
        # 4. 高度偏移
        if random.random() < self.altitude_shift_prob:
            augmented_data = self._apply_altitude_shift(augmented_data)
        
        # 5. 高斯噪声
        if random.random() < self.gaussian_noise_prob:
            augmented_data = self._apply_gaussian_noise(augmented_data)
        
        # 6. 时间缩放
        if random.random() < self.time_scaling_prob:
            augmented_data = self._apply_time_scaling(augmented_data)
        
        return augmented_data
    
    def _apply_trajectory_warp(self, data: torch.Tensor) -> torch.Tensor:
        """应用轨迹扭曲（影响x, y, z位置）"""
        augmented_data = data.clone()
        
        if data.shape[1] >= 3:  # 确保有x, y, z特征
            seq_len = data.shape[0]
            
            # 为每个空间维度生成平滑的扭曲
            for dim in range(3):  # x, y, z
                # 生成随机控制点
                num_control_points = random.randint(2, 4)
                control_indices = np.linspace(0, seq_len-1, num_control_points, dtype=int)
                control_values = np.random.uniform(-self.warp_magnitude, 
                                                 self.warp_magnitude, 
                                                 num_control_points)
                
                # 插值生成完整的扭曲序列
                warp = np.interp(np.arange(seq_len), control_indices, control_values)
                
                # 应用扭曲
                position_std = torch.std(data[:, dim])
                augmented_data[:, dim] += torch.tensor(warp, dtype=torch.float32) * position_std
        
        return augmented_data
    
    def _apply_attitude_noise(self, data: torch.Tensor) -> torch.Tensor:
        """应用姿态噪声（影响roll, pitch, yaw）"""
        augmented_data = data.clone()
        
        if data.shape[1] >= 6:  # 确保有roll, pitch, yaw特征
            # 使用固定std值，可以使用size参数
            attitude_noise = torch.normal(mean=0.0, std=self.attitude_noise_std, size=(data.shape[0], 3))
            augmented_data[:, 3:6] += attitude_noise
        
        return augmented_data
    
    def _apply_velocity_perturbation(self, data: torch.Tensor) -> torch.Tensor:
        """应用速度扰动"""
        augmented_data = data.clone()
        
        if data.shape[1] > 6:  # 确保有velocity特征
            # 使用固定std值，可以使用size参数
            velocity_noise = torch.normal(mean=0.0, std=self.velocity_perturb_ratio, size=(data.shape[0], 1))
            velocity_std = torch.std(data[:, 6])
            augmented_data[:, 6:7] += velocity_noise * velocity_std
        
        return augmented_data
    
    def _apply_altitude_shift(self, data: torch.Tensor) -> torch.Tensor:
        """应用高度偏移"""
        augmented_data = data.clone()
        
        if data.shape[1] > 8:  # 确保有altitude特征
            altitude_shift = random.uniform(-self.altitude_shift_range, self.altitude_shift_range)
            augmented_data[:, 8] += altitude_shift
        
        return augmented_data
    
    def _apply_gaussian_noise(self, data: torch.Tensor) -> torch.Tensor:
        """应用高斯噪声"""
        # 计算每个特征的标准差
        feature_stds = torch.std(data, dim=0, keepdim=True)
        
        # 生成噪声 - 当std是Tensor时，不能使用size参数
        # 需要广播std到正确的形状
        std_tensor = self.noise_std_ratio * feature_stds.expand_as(data)
        noise = torch.normal(mean=0.0, std=std_tensor)
        
        return data + noise
    
    def _apply_time_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """应用时间缩放"""
        seq_len, num_features = data.shape
        
        # 生成缩放因子
        scale_factor = random.uniform(*self.time_scaling_range)
        
        # 生成新的时间索引
        original_indices = torch.arange(seq_len, dtype=torch.float32)
        scaled_indices = original_indices * scale_factor
        
        # 确保索引在有效范围内
        scaled_indices = torch.clamp(scaled_indices, 0, seq_len - 1)
        
        # 应用插值
        augmented_data = torch.zeros_like(data)
        for i in range(num_features):
            # 使用线性插值
            augmented_data[:, i] = torch.nn.functional.interpolate(
                data[:, i].unsqueeze(0).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return augmented_data