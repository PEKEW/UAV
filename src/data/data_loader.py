"""
高效的双域数据加载器
支持GPU优化、异步加载和内存管理
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc

from .preprocessor import BatteryPreprocessor, FlightPreprocessor
from .augmentation import BatteryAugmentation, FlightAugmentation


class BaseDataset(Dataset):
    """数据集基类"""
    
    def __init__(self, data_path: str, preprocessor, augmentation=None, 
                 sequence_length: int = 30, train_mode: bool = True):
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.sequence_length = sequence_length
        self.train_mode = train_mode
        
        # 加载和预处理数据
        self._load_data()
        self._prepare_samples()
        
    def _load_data(self):
        """加载数据"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        print(f"加载数据: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
    def _prepare_samples(self):
        """准备样本数据"""
        # 获取唯一样本ID
        if 'sample_id' in self.raw_data.columns:
            self.sample_ids = self.raw_data['sample_id'].unique()
        else:
            # 如果没有sample_id，按sequence_length切分
            total_rows = len(self.raw_data)
            self.sample_ids = list(range(0, total_rows, self.sequence_length))
        
        print(f"总样本数: {len(self.sample_ids)}")
        
        # 预处理所有样本
        self._preprocess_all_samples()
        
    def _preprocess_all_samples(self):
        """预处理所有样本"""
        self.samples = []
        self.labels = []
        
        # 使用所有可用的样本ID（样本限制已在数据分割时处理）
        print(f"处理样本数量: {len(self.sample_ids)}")
        
        processed_count = 0
        failed_count = 0
        
        for i, sample_id in enumerate(self.sample_ids):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(self.sample_ids)} ({i/len(self.sample_ids)*100:.1f}%)")
            
            try:
                sample_data, label = self._get_sample_data(sample_id)
                if sample_data is not None:
                    # 预处理
                    processed_data = self.preprocessor.transform(sample_data)
                    
                    # 验证数据形状
                    if processed_data.shape[0] != self.sequence_length:
                        print(f"警告: 样本 {sample_id} 序列长度不正确: {processed_data.shape}")
                        continue
                    
                    self.samples.append(processed_data)
                    self.labels.append(label)
                    processed_count += 1
                else:
                    failed_count += 1
                    if failed_count <= 10:  # 只显示前10个失败的详情
                        print(f"样本 {sample_id} 数据为空或不满足条件")
                        
            except Exception as e:
                failed_count += 1
                if failed_count <= 10:  # 只显示前10个错误的详情
                    print(f"样本 {sample_id} 处理失败: {str(e)}")
                continue
        
        print(f"预处理完成: 成功 {processed_count}, 失败 {failed_count}")
        
        if len(self.samples) == 0:
            raise ValueError("没有成功处理的样本！请检查数据格式和处理逻辑。")
        
        # 转换为tensor
        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"最终数据形状: samples={self.samples.shape}, labels={self.labels.shape}")
        
        # 计算类别权重
        self._compute_class_weights()
    def _get_sample_data(self, sample_id) -> Tuple[Optional[torch.Tensor], int]:
        """获取单个样本数据，由子类实现"""
        raise NotImplementedError
        
    def _compute_class_weights(self):
        """计算类别权重用于平衡采样"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        self.class_weights = torch.zeros(2)  # 假设二分类
        for label, count in zip(unique_labels, counts):
            self.class_weights[label] = total / (len(unique_labels) * count)
        
        print(f"类别分布: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        print(f"类别权重: {self.class_weights.tolist()}")
        
    def get_sample_weights(self):
        """获取样本权重用于WeightedRandomSampler"""
        return self.class_weights[self.labels]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].clone()  # 避免修改原始数据
        label = self.labels[idx]
        
        # 训练时应用数据增强
        if self.train_mode and self.augmentation is not None:
            sample = self.augmentation(sample)
            
        return sample, label


class H5Dataset(Dataset):
    """H5数据集类 - 专门用于加载3D格式的H5文件"""
    
    def __init__(self, h5_path: str, preprocessor=None, augmentation=None, train_mode: bool = True):
        self.h5_path = Path(h5_path)
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.train_mode = train_mode
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5数据文件不存在: {self.h5_path}")
        
        self._load_h5_data()
        self._prepare_data()
        
    def _load_h5_data(self):
        """加载H5数据"""
        print(f"加载H5数据: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            # 加载数据
            self.data = torch.tensor(f['data'][:], dtype=torch.float32)
            self.labels = torch.tensor(f['labels'][:], dtype=torch.long)
            
            # 加载特征名称
            if 'feature_names' in f:
                feature_names_bytes = f['feature_names'][:]
                self.feature_names = [name.decode() if isinstance(name, bytes) else name 
                                    for name in feature_names_bytes]
            else:
                self.feature_names = [f'feature_{i}' for i in range(self.data.shape[2])]
        
        print(f"H5数据加载完成:")
        print(f"  数据形状: {self.data.shape}")
        print(f"  标签形状: {self.labels.shape}")
        print(f"  特征名称: {self.feature_names}")
        
        # 验证数据完整性
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError(f"数据和标签样本数不匹配: {self.data.shape[0]} vs {self.labels.shape[0]}")
            
    def _prepare_data(self):
        """准备数据"""
        # 检查数据中的异常值
        if torch.isnan(self.data).any():
            print("警告: 数据中包含NaN值，将被替换为0")
            self.data = torch.nan_to_num(self.data, nan=0.0)
            
        if torch.isinf(self.data).any():
            print("警告: 数据中包含无穷大值，将被替换为0")
            self.data = torch.nan_to_num(self.data, posinf=0.0, neginf=0.0)
        
        # 应用预处理器
        if self.preprocessor is not None:
            print("应用预处理器...")
            # 重塑数据以适应预处理器 (samples * timesteps, features)
            original_shape = self.data.shape
            reshaped_data = self.data.view(-1, original_shape[2])
            
            # 转换为numpy进行预处理
            data_np = reshaped_data.numpy()
            
            # 拟合并转换数据
            if self.train_mode:
                processed_data = self.preprocessor.fit_transform(data_np)
            else:
                processed_data = self.preprocessor.transform(data_np)
            
            # 确保processed_data是numpy数组
            if isinstance(processed_data, torch.Tensor):
                processed_data = processed_data.numpy()
            
            # 转换回tensor并重塑为原始3D形状
            self.data = torch.from_numpy(processed_data).float().view(original_shape)
        
        # 计算类别权重
        self._compute_class_weights()
        
        print(f"数据准备完成: {len(self)} 个样本")
        
    def _compute_class_weights(self):
        """计算类别权重用于平衡采样"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        # 假设二分类，创建权重tensor
        self.class_weights = torch.zeros(2)
        for label, count in zip(unique_labels, counts):
            self.class_weights[label] = total / (len(unique_labels) * count)
        
        print(f"类别分布: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        print(f"类别权重: {self.class_weights.tolist()}")
        
    def get_sample_weights(self):
        """获取样本权重用于WeightedRandomSampler"""
        return self.class_weights[self.labels]
    
    @property
    def samples(self):
        """为了兼容性，提供samples属性"""
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].clone()  # 避免修改原始数据
        label = self.labels[idx]
        
        # 训练时应用数据增强
        if self.train_mode and self.augmentation is not None:
            sample = self.augmentation(sample)
            
        return sample, label


class BatteryDataset(BaseDataset):
    """电池数据集"""
    
    def __init__(self, data_path: str, sequence_length: int = 30, 
                 train_mode: bool = True, augmentation_config: Optional[Dict] = None):
        
        # 初始化预处理器和增强器
        preprocessor = BatteryPreprocessor()
        augmentation = BatteryAugmentation(augmentation_config) if augmentation_config else None
        
        super().__init__(data_path, preprocessor, augmentation, sequence_length, train_mode)
        
    def _get_sample_data(self, sample_id) -> Tuple[Optional[torch.Tensor], int]:
        """获取电池样本数据"""
        try:
            if 'sample_id' in self.raw_data.columns:
                sample_data = self.raw_data[self.raw_data['sample_id'] == sample_id]
            else:
                # 基于索引切分
                start_idx = sample_id
                end_idx = min(start_idx + self.sequence_length, len(self.raw_data))
                sample_data = self.raw_data.iloc[start_idx:end_idx]
            
            if len(sample_data) == 0:
                return None, 0
                
            # 提取特征和标签
            feature_columns = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                              'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
            
            # 检查必要列是否存在
            available_features = [col for col in feature_columns if col in sample_data.columns]
            if len(available_features) == 0:
                return None, 0
            
            # 对于有sample_id的数据，确保我们有足够的行数
            if 'sample_id' in self.raw_data.columns:
                if len(sample_data) < self.sequence_length:
                    # 如果一个sample_id对应的数据行数不足30行，跳过
                    return None, 0
                elif len(sample_data) > self.sequence_length:
                    # 如果超过30行，取前30行
                    sample_data = sample_data.head(self.sequence_length)
            else:
                # 基于索引的情况下，确保序列长度
                if len(sample_data) < self.sequence_length:
                    return None, 0
                
            # 提取特征数据
            features = sample_data[available_features].values
            
            # 验证数据形状
            if features.shape[0] != self.sequence_length:
                return None, 0
            
            if features.shape[1] != len(available_features):
                return None, 0
            
            # 处理缺失值
            if np.isnan(features).any():
                features = np.nan_to_num(features, nan=0.0)
            
            # 检查是否有无穷大值
            if np.isinf(features).any():
                features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
            
            # 获取标签
            label = 0  # 默认正常样本
            if 'label' in sample_data.columns:
                label_values = sample_data['label'].unique()
                if len(label_values) > 0:
                    label = int(label_values[0])
            elif 'is_anomaly' in sample_data.columns:
                is_anomaly_values = sample_data['is_anomaly'].unique()
                if len(is_anomaly_values) > 0:
                    # 处理布尔值或字符串
                    first_value = is_anomaly_values[0]
                    if isinstance(first_value, bool):
                        label = int(first_value)
                    elif isinstance(first_value, str):
                        label = 1 if first_value.lower() == 'true' else 0
                    else:
                        label = int(first_value)
                
            return torch.tensor(features, dtype=torch.float32), label
            
        except Exception as e:
            # 提供更详细的错误信息
            print(f"处理样本 {sample_id} 时发生错误: {str(e)}")
            if hasattr(self, 'raw_data') and 'sample_id' in self.raw_data.columns:
                sample_data = self.raw_data[self.raw_data['sample_id'] == sample_id]
                print(f"样本数据形状: {sample_data.shape}")
                if len(sample_data) > 0:
                    print(f"样本数据列: {list(sample_data.columns)}")
            return None, 0


class FlightDataset(BaseDataset):
    """飞行数据集"""
    
    def __init__(self, data_path: str, sequence_length: int = 30, 
                 train_mode: bool = True, augmentation_config: Optional[Dict] = None):
        
        # 初始化预处理器和增强器
        preprocessor = FlightPreprocessor()
        augmentation = FlightAugmentation(augmentation_config) if augmentation_config else None
        
        super().__init__(data_path, preprocessor, augmentation, sequence_length, train_mode)
        
    def _get_sample_data(self, sample_id) -> Tuple[Optional[torch.Tensor], int]:
        """获取飞行样本数据"""
        try:
            if 'sample_id' in self.raw_data.columns:
                sample_data = self.raw_data[self.raw_data['sample_id'] == sample_id]
            else:
                # 基于索引切分
                start_idx = sample_id
                end_idx = min(start_idx + self.sequence_length, len(self.raw_data))
                sample_data = self.raw_data.iloc[start_idx:end_idx]
            
            if len(sample_data) == 0:
                return None, 0
                
            # 提取特征和标签
            feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                              'velocity', 'acceleration', 'altitude']
            
            # 检查必要列是否存在
            available_features = [col for col in feature_columns if col in sample_data.columns]
            if len(available_features) == 0:
                return None, 0
            
            # 对于有sample_id的数据，确保我们有足够的行数
            if 'sample_id' in self.raw_data.columns:
                if len(sample_data) < self.sequence_length:
                    # 如果一个sample_id对应的数据行数不足30行，跳过
                    return None, 0
                elif len(sample_data) > self.sequence_length:
                    # 如果超过30行，取前30行
                    sample_data = sample_data.head(self.sequence_length)
            else:
                # 基于索引的情况下，确保序列长度
                if len(sample_data) < self.sequence_length:
                    return None, 0
                
            # 提取特征数据
            features = sample_data[available_features].values
            
            # 验证数据形状
            if features.shape[0] != self.sequence_length:
                return None, 0
            
            if features.shape[1] != len(available_features):
                return None, 0
            
            # 处理缺失值
            if np.isnan(features).any():
                features = np.nan_to_num(features, nan=0.0)
            
            # 检查是否有无穷大值
            if np.isinf(features).any():
                features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
            
            # 获取标签
            label = 0  # 默认正常样本
            if 'label' in sample_data.columns:
                label_values = sample_data['label'].unique()
                if len(label_values) > 0:
                    label = int(label_values[0])
            elif 'is_anomaly' in sample_data.columns:
                is_anomaly_values = sample_data['is_anomaly'].unique()
                if len(is_anomaly_values) > 0:
                    # 处理布尔值或字符串
                    first_value = is_anomaly_values[0]
                    if isinstance(first_value, bool):
                        label = int(first_value)
                    elif isinstance(first_value, str):
                        label = 1 if first_value.lower() == 'true' else 0
                    else:
                        label = int(first_value)
                
            return torch.tensor(features, dtype=torch.float32), label
            
        except Exception as e:
            # 提供更详细的错误信息
            print(f"处理飞行样本 {sample_id} 时发生错误: {str(e)}")
            if hasattr(self, 'raw_data') and 'sample_id' in self.raw_data.columns:
                sample_data = self.raw_data[self.raw_data['sample_id'] == sample_id]
                print(f"飞行样本数据形状: {sample_data.shape}")
                if len(sample_data) > 0:
                    print(f"飞行样本数据列: {list(sample_data.columns)}")
            return None, 0


class BatteryH5Dataset(H5Dataset):
    """电池H5数据集类"""
    
    def __init__(self, h5_path: str, train_mode: bool = True, augmentation_config: Optional[Dict] = None):
        # 初始化预处理器和增强器
        preprocessor = BatteryPreprocessor()
        augmentation = BatteryAugmentation(augmentation_config) if augmentation_config else None
        
        super().__init__(h5_path, preprocessor, augmentation, train_mode)
        
        # 验证特征名称是否符合电池数据格式
        expected_features = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                           'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
        
        if len(self.feature_names) != len(expected_features):
            print(f"警告: 特征数量不匹配，期望 {len(expected_features)}，实际 {len(self.feature_names)}")
        
        # 验证特征名称
        for i, (expected, actual) in enumerate(zip(expected_features, self.feature_names)):
            if expected != actual:
                print(f"警告: 第{i}个特征名称不匹配，期望 '{expected}'，实际 '{actual}'")


class FlightH5Dataset(H5Dataset):
    """飞行H5数据集类"""
    
    def __init__(self, h5_path: str, train_mode: bool = True, augmentation_config: Optional[Dict] = None):
        # 初始化预处理器和增强器
        preprocessor = FlightPreprocessor()
        augmentation = FlightAugmentation(augmentation_config) if augmentation_config else None
        
        super().__init__(h5_path, preprocessor, augmentation, train_mode)
        
        # 验证特征名称是否符合飞行数据格式
        expected_features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                           'velocity', 'acceleration', 'altitude']
        
        if len(self.feature_names) != len(expected_features):
            print(f"警告: 特征数量不匹配，期望 {len(expected_features)}，实际 {len(self.feature_names)}")


class DualDomainDataLoader:
    """双域数据加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 8)
        self.pin_memory = config.get('pin_memory', True)
        self.prefetch_factor = config.get('prefetch_factor', 4)
        self.persistent_workers = config.get('persistent_workers', True)
        
        # 数据路径
        self.battery_data_path = config.get('battery_data_path', 'data/evtol_anomaly_dataset.csv')
        self.flight_data_path = config.get('flight_data_path', 'data/flight_anomaly_dataset.csv')
        self.battery_h5_path = config.get('battery_h5_path', 'processed/evtol_anomaly_dataset_3d.h5')
        self.flight_h5_path = config.get('flight_h5_path', 'processed/flight_anomaly_dataset_3d.h5')
        
        # 增强配置
        self.battery_aug_config = config.get('battery_augmentation', {})
        self.flight_aug_config = config.get('flight_augmentation', {})
        
        # 初始化数据集
        self._create_datasets()
        
    def _create_split_datasets(self, data_path: str, train_split: float = 0.8, 
                              augmentation_config: Optional[Dict] = None, dataset_type: str = 'battery'):
        """创建训练和验证数据集的分割版本"""
        # 首先读取数据以获取样本ID
        raw_data = pd.read_csv(data_path)
        
        if 'sample_id' in raw_data.columns:
            unique_sample_ids = raw_data['sample_id'].unique()
        else:
            # 如果没有sample_id，基于行数创建
            sequence_length = 30
            total_rows = len(raw_data)
            unique_sample_ids = list(range(0, total_rows, sequence_length))
        
        # 应用样本限制BEFORE分割，确保我们有足够的数据进行有意义的训练
        max_total_samples = 20000  # 总样本限制
        if len(unique_sample_ids) > max_total_samples:
            print(f"限制样本数量从 {len(unique_sample_ids)} 到 {max_total_samples}")
            # 随机选择样本以保持数据分布
            np.random.seed(42)
            selected_indices = np.random.choice(len(unique_sample_ids), max_total_samples, replace=False)
            unique_sample_ids = unique_sample_ids[selected_indices]
        
        # 打乱样本ID以确保随机分割
        np.random.seed(42)  # 确保可重现性
        shuffled_ids = np.random.permutation(unique_sample_ids)
        
        # 分割样本ID
        n_train = int(len(shuffled_ids) * train_split)
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:]
        
        print(f"数据分割: 训练样本 {len(train_ids)}, 验证样本 {len(val_ids)}")
        
        # 创建训练和验证数据文件
        if 'sample_id' in raw_data.columns:
            train_data = raw_data[raw_data['sample_id'].isin(train_ids)]
            val_data = raw_data[raw_data['sample_id'].isin(val_ids)]
        else:
            # 基于索引分割
            train_indices = []
            val_indices = []
            sequence_length = 30
            
            for sample_id in train_ids:
                start_idx = sample_id
                end_idx = min(start_idx + sequence_length, len(raw_data))
                train_indices.extend(range(start_idx, end_idx))
            
            for sample_id in val_ids:
                start_idx = sample_id
                end_idx = min(start_idx + sequence_length, len(raw_data))
                val_indices.extend(range(start_idx, end_idx))
            
            train_data = raw_data.iloc[train_indices]
            val_data = raw_data.iloc[val_indices]
        
        # 创建临时文件
        import tempfile
        train_temp_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False)
        val_temp_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False)
        
        train_data.to_csv(train_temp_path.name, index=False)
        val_data.to_csv(val_temp_path.name, index=False)
        
        # 创建数据集对象
        if dataset_type == 'battery':
            train_dataset = BatteryDataset(
                train_temp_path.name,
                train_mode=True,
                augmentation_config=augmentation_config
            )
            val_dataset = BatteryDataset(
                val_temp_path.name,
                train_mode=False
            )
        else:  # flight
            train_dataset = FlightDataset(
                train_temp_path.name,
                train_mode=True,
                augmentation_config=augmentation_config
            )
            val_dataset = FlightDataset(
                val_temp_path.name,
                train_mode=False
            )
        
        # 验证数据集大小
        print(f"最终数据集大小: 训练 {len(train_dataset)} 样本, 验证 {len(val_dataset)} 样本")
        
        if len(val_dataset) == 0:
            raise ValueError("验证数据集为空！请检查数据分割和预处理逻辑。")
        
        if len(train_dataset) == 0:
            raise ValueError("训练数据集为空！请检查数据分割和预处理逻辑。")
        
        # 清理临时文件
        import os
        os.unlink(train_temp_path.name)
        os.unlink(val_temp_path.name)
        
        return train_dataset, val_dataset
    
    def _create_h5_split_datasets(self, h5_path: str, train_split: float = 0.8, 
                                 augmentation_config: Optional[Dict] = None, dataset_type: str = 'battery'):
        """创建H5数据集的训练和验证分割"""
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"H5文件不存在: {h5_path}")
        
        print(f"加载H5数据: {h5_path}")
        
        # 首先加载完整数据集以获取样本数量
        with h5py.File(h5_path, 'r') as f:
            total_samples = f['data'].shape[0]
            print(f"H5文件总样本数: {total_samples}")
        
        # 应用样本限制
        max_total_samples = 20000
        if total_samples > max_total_samples:
            print(f"限制样本数量从 {total_samples} 到 {max_total_samples}")
            total_samples = max_total_samples
        
        # 创建样本索引并打乱
        np.random.seed(42)
        indices = np.random.permutation(total_samples)
        
        # 分割索引
        n_train = int(len(indices) * train_split)
        train_indices = sorted(indices[:n_train])
        val_indices = sorted(indices[n_train:])
        
        print(f"H5数据分割: 训练样本 {len(train_indices)}, 验证样本 {len(val_indices)}")
        
        # 创建临时H5文件
        import tempfile
        train_temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        val_temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        
        # 复制数据到临时文件
        with h5py.File(h5_path, 'r') as source:
            # 创建训练数据文件
            with h5py.File(train_temp_file.name, 'w') as train_f:
                train_f.create_dataset('data', data=source['data'][train_indices])
                train_f.create_dataset('labels', data=source['labels'][train_indices])
                if 'feature_names' in source:
                    train_f.create_dataset('feature_names', data=source['feature_names'][:])
            
            # 创建验证数据文件
            with h5py.File(val_temp_file.name, 'w') as val_f:
                val_f.create_dataset('data', data=source['data'][val_indices])
                val_f.create_dataset('labels', data=source['labels'][val_indices])
                if 'feature_names' in source:
                    val_f.create_dataset('feature_names', data=source['feature_names'][:])
        
        # 创建数据集对象
        if dataset_type == 'battery':
            train_dataset = BatteryH5Dataset(
                train_temp_file.name,
                train_mode=True,
                augmentation_config=augmentation_config
            )
            val_dataset = BatteryH5Dataset(
                val_temp_file.name,
                train_mode=False
            )
        else:  # flight
            train_dataset = FlightH5Dataset(
                train_temp_file.name,
                train_mode=True,
                augmentation_config=augmentation_config
            )
            val_dataset = FlightH5Dataset(
                val_temp_file.name,
                train_mode=False
            )
        
        # 验证数据集大小
        print(f"最终H5数据集大小: 训练 {len(train_dataset)} 样本, 验证 {len(val_dataset)} 样本")
        
        if len(val_dataset) == 0:
            raise ValueError("验证数据集为空！")
        
        if len(train_dataset) == 0:
            raise ValueError("训练数据集为空！")
        
        # 清理临时文件
        import os
        os.unlink(train_temp_file.name)
        os.unlink(val_temp_file.name)
        
        return train_dataset, val_dataset
        
    def _create_datasets(self):
        """创建数据集"""
        print("创建数据集...")
        
        # 电池数据集 - 优先使用H5文件
        if Path(self.battery_h5_path).exists():
            print("使用H5格式的电池数据集")
            self.battery_train_dataset, self.battery_val_dataset = self._create_h5_split_datasets(
                self.battery_h5_path,
                train_split=0.8,
                augmentation_config=self.battery_aug_config,
                dataset_type='battery'
            )
            print(f"电池H5数据集: 训练 {len(self.battery_train_dataset)} 样本, 验证 {len(self.battery_val_dataset)} 样本")
        elif Path(self.battery_data_path).exists():
            print("使用CSV格式的电池数据集")
            self.battery_train_dataset, self.battery_val_dataset = self._create_split_datasets(
                self.battery_data_path, 
                train_split=0.8,
                augmentation_config=self.battery_aug_config
            )
            print(f"电池CSV数据集: 训练 {len(self.battery_train_dataset)} 样本, 验证 {len(self.battery_val_dataset)} 样本")
        else:
            print(f"警告: 电池数据文件不存在 - H5: {self.battery_h5_path}, CSV: {self.battery_data_path}")
            self.battery_train_dataset = None
            self.battery_val_dataset = None
        
        # 飞行数据集 - 优先使用H5文件
        if Path(self.flight_h5_path).exists():
            print("使用H5格式的飞行数据集")
            self.flight_train_dataset, self.flight_val_dataset = self._create_h5_split_datasets(
                self.flight_h5_path,
                train_split=0.8,
                augmentation_config=self.flight_aug_config,
                dataset_type='flight'
            )
            print(f"飞行H5数据集: 训练 {len(self.flight_train_dataset)} 样本, 验证 {len(self.flight_val_dataset)} 样本")
        else:
            raise FileNotFoundError(f"飞行数据文件不存在 - H5: {self.flight_h5_path}")
    
    def get_battery_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """获取电池数据加载器"""
        if self.battery_train_dataset is None:
            return None, None
            
        # 创建加权采样器
        sample_weights = self.battery_train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            self.battery_train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.battery_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
        
        return train_loader, val_loader
    
    def get_flight_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """获取飞行数据加载器"""
        if self.flight_train_dataset is None:
            return None, None
            
        # 创建加权采样器
        sample_weights = self.flight_train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            self.flight_train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.flight_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
        
        return train_loader, val_loader
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        info = {
            'battery_dataset': None,
            'flight_dataset': None
        }
        
        if self.battery_train_dataset is not None:
            info['battery_dataset'] = {
                'train_samples': len(self.battery_train_dataset),
                'val_samples': len(self.battery_val_dataset),
                'input_shape': self.battery_train_dataset.samples.shape[1:],
                'class_distribution': self.battery_train_dataset.class_weights.tolist()
            }
        
        if self.flight_train_dataset is not None:
            info['flight_dataset'] = {
                'train_samples': len(self.flight_train_dataset),
                'val_samples': len(self.flight_val_dataset),
                'input_shape': self.flight_train_dataset.samples.shape[1:],
                'class_distribution': self.flight_train_dataset.class_weights.tolist()
            }
            
        return info
    
    def cleanup(self):
        """清理资源"""
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()