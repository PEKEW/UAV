import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
from pathlib import Path
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing
from functools import lru_cache

feature_columns = [
    'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
    'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C',
    'cycleNumber', 'time_s'
]

scale_columns = [
    'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
    'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C'
]
    


class BatteryAnomalyDataset(Dataset):
    def __init__(self, data_path: str, sequence_length: int = 50, 
                transform=None, is_train: bool = True, file_ids: List[str] = None,
                classification_mode: bool = True):
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_train = is_train
        self.data_path = data_path
        self.classification_mode = classification_mode
        
        self.feature_columns = [
            'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
            'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C'
        ]
        self.file_ids = file_ids if file_ids else ['battery_anomaly_dataset']
        
        self._preload_data()
        
        # 计算序列数并创建索引映射
        self.sequence_indices = []
        self.total_sequences = 0
        
        self._initialize_sequences()
        
        if self.total_sequences == 0:
            raise ValueError("No valid sequences found in the dataset!")
    
    def _preload_data(self):
        print("预加载电池异常检测数据...")
        self.data_cache = {}
        
        for file_id in self.file_ids:
            file_path = Path(self.data_path) / f"{file_id}.csv"
            if not file_path.exists():
                continue
                
            df = pd.read_csv(file_path)
            self.data_cache[file_id] = df
    
    def _initialize_sequences(self):
        for file_id, df in self.data_cache.items():
            n_samples = len(df) - self.sequence_length + 1
            
            if n_samples > 0:
                for i in range(n_samples):
                    self.sequence_indices.append({
                        'file_id': file_id,
                        'start_idx': i,
                        'end_idx': i + self.sequence_length
                    })
                self.total_sequences += n_samples
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_info = self.sequence_indices[idx]
        df = self.data_cache[seq_info['file_id']]
        
        sequence_data = df.iloc[seq_info['start_idx']:seq_info['end_idx']]
        sequence = sequence_data[self.feature_columns].values
        
        if self.classification_mode:
            # 分类模式：返回序列的最后一个时间步的标签
            label = sequence_data['label'].iloc[-1]
            sequence = torch.FloatTensor(sequence)
            label = torch.LongTensor([label])
            return sequence, label
        else:
            # 回归模式：预测下一个时间步
            if seq_info['end_idx'] < len(df):
                target = df.iloc[seq_info['end_idx']]['Ecell_V']
                sequence = torch.FloatTensor(sequence)
                target = torch.FloatTensor([target])
                return sequence, target
            else:
                # 如果没有下一个时间步，返回当前时间步
                target = sequence_data['Ecell_V'].iloc[-1]
                sequence = torch.FloatTensor(sequence)
                target = torch.FloatTensor([target])
                return sequence, target

class EVTOLDataset(Dataset):
    def __init__(self, data_path: str, sequence_length: int = 100, prediction_steps: int = 1, 
                transform=None, is_train: bool = True, file_ids: List[str] = None):
        self.anomalies = {
            "VAH05": [1000],
            "VAH09": [64, 92, 154, 691],
            "VAH10": [248, 631, 735, 1151],
            "VAH11": [817, 1898],
            "VAH13": [816, 817],
            "VAH25": [461, 462],
            "VAH26": [872, 873],
            "VAH27": [20, 256, 257, 585],
            "VAH28": [256, 257, 619, 620, 1066, 1067],
        }
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.transform = transform
        self.is_train = is_train
        self.data_path = data_path
        
        self.feature_columns = feature_columns
        self.scale_columns = scale_columns
        # 默认预测目标为电池电压
        self.prediction_targets = ["Ecell_V"]
        self.file_ids = file_ids
        
        self._preload_data()
        
        # 计算每个文件的序列数并创建索引映射
        self.sequence_indices = []
        self.total_sequences = 0
        # 存储每个文件的cycle信息
        self.file_cycle_info = {}  
        
        self._initialize_file_info()
        
        if self.total_sequences == 0:
            raise ValueError("No valid sequences found in the dataset!")
    
    def _preload_data(self):
        print("预加载数据...")
        self.data_cache = {}
        
        for file_id in self.file_ids:
            matching_files = list(Path(self.data_path).glob(f"*{file_id}*.csv"))
            if not matching_files:
                continue
                
            file_path = matching_files[0]
            df = pd.read_csv(file_path)
            
            df['time_s'] = df['time_s'] / df['time_s'].max()
            for cycle in df['cycleNumber'].unique():
                cycle_data = df[df['cycleNumber'] == cycle].copy()
                self.data_cache[(file_id, cycle)] = cycle_data
    
    def _initialize_file_info(self):
        for (file_id, cycle), cycle_data in self.data_cache.items():
            if file_id in self.anomalies and cycle in self.anomalies[file_id]:
                continue
                
            n_samples = len(cycle_data) - self.sequence_length - self.prediction_steps + 1
            
            if n_samples > 0:
                if file_id not in self.file_cycle_info:
                    self.file_cycle_info[file_id] = {
                        'cycles': [],
                        'sequence_counts': {}
                    }
                
                self.file_cycle_info[file_id]['cycles'].append(cycle)
                self.file_cycle_info[file_id]['sequence_counts'][cycle] = n_samples
                
                for i in range(n_samples):
                    self.sequence_indices.append({
                        'file_id': file_id,
                        'cycle': cycle,
                        'start_idx': i,
                        'end_idx': i + self.sequence_length,
                        'target_idx': i + self.sequence_length
                    })
                self.total_sequences += n_samples
        
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_info = self.sequence_indices[idx]
        cycle_data = self.data_cache[(seq_info['file_id'], seq_info['cycle'])]
        sequence = cycle_data.iloc[seq_info['start_idx']:seq_info['end_idx']][self.feature_columns].values
        target = cycle_data.iloc[seq_info['target_idx']:seq_info['target_idx'] + self.prediction_steps][self.feature_columns].values
        sequence = torch.FloatTensor(sequence)
        target_idx = self.feature_columns.index(self.prediction_targets[0])
        target = torch.FloatTensor(target[:, target_idx:target_idx+1])  # [prediction_steps, 1]
        if self.transform:
            sequence = self.transform(sequence)
            target = self.transform(target)
        return sequence, target

def get_battery_dataset(data_path: str, sequence_length: int = 50,
                       train_ratio: float = 0.7, val_ratio: float = 0.15,
                       classification_mode: bool = True) -> Tuple[BatteryAnomalyDataset, BatteryAnomalyDataset, BatteryAnomalyDataset]:
    """获取电池异常检测数据集"""
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"数据路径不存在: {data_path}")
    
    # 加载完整数据集
    file_path = data_path / "battery_anomaly_dataset.csv"
    if not file_path.exists():
        raise ValueError(f"电池数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    n_samples = len(df)
    
    # 按时间序列划分数据集
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 保存分割后的数据
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    train_df.to_csv(data_path / "battery_train.csv", index=False)
    val_df.to_csv(data_path / "battery_val.csv", index=False)
    test_df.to_csv(data_path / "battery_test.csv", index=False)
    
    # 创建数据集对象
    train_dataset = BatteryAnomalyDataset(str(data_path), sequence_length, 
                                         file_ids=['battery_train'], 
                                         classification_mode=classification_mode)
    val_dataset = BatteryAnomalyDataset(str(data_path), sequence_length, 
                                       file_ids=['battery_val'], 
                                       classification_mode=classification_mode)
    test_dataset = BatteryAnomalyDataset(str(data_path), sequence_length, 
                                        file_ids=['battery_test'], 
                                        classification_mode=classification_mode)
    
    return train_dataset, val_dataset, test_dataset

def get_dataset(data_path: str, sequence_length: int = 50, prediction_steps: int = 1,
                train_ratio: float = 0.7, val_ratio: float = 0.15,
                prediction_targets: List[str] = None) -> Tuple[EVTOLDataset, EVTOLDataset, EVTOLDataset]:
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"路径不存在")
        
    files = list(data_path.glob('*.csv'))
    
    if not files:
        raise ValueError(f"路径中不存在csv文件")
    
    file_ids = []
    for file in files:
        file_id = file.stem
        file_ids.append(file_id)
        # print(f"\n处理文件: {file}")
        df = pd.read_csv(file)
        # print(f"文件行数: {len(df)}")
    
    
    if len(file_ids) == 1:
        train_dataset = EVTOLDataset(data_path, sequence_length, prediction_steps, file_ids=file_ids)
        return train_dataset, train_dataset, train_dataset
    
    np.random.shuffle(file_ids)
    n_files = len(file_ids)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = file_ids[:n_train]
    val_files = file_ids[n_train:n_train + n_val]
    test_files = file_ids[n_train + n_val:]
    
    train_dataset = EVTOLDataset(data_path, sequence_length, prediction_steps, file_ids=train_files)
    val_dataset = EVTOLDataset(data_path, sequence_length, prediction_steps, file_ids=val_files)
    test_dataset = EVTOLDataset(data_path, sequence_length, prediction_steps, file_ids=test_files)
    
    if prediction_targets:
        train_dataset.prediction_targets = prediction_targets
        val_dataset.prediction_targets = prediction_targets
        test_dataset.prediction_targets = prediction_targets
    
    return train_dataset, val_dataset, test_dataset

class FlightAnomalyDataset(Dataset):
    def __init__(self, data_path: str, sequence_length: int = 50, 
                transform=None, is_train: bool = True, file_ids: List[str] = None,
                classification_mode: bool = True):
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_train = is_train
        self.data_path = data_path
        self.classification_mode = classification_mode
        
        self.feature_columns = [
            'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude'
        ]
        self.file_ids = file_ids if file_ids else ['flight_attitude_anomaly_dataset']
        
        self._preload_data()
        
        # 计算序列数并创建索引映射
        self.sequence_indices = []
        self.total_sequences = 0
        
        self._initialize_sequences()
        
        if self.total_sequences == 0:
            raise ValueError("No valid sequences found in the dataset!")
    
    def _preload_data(self):
        print("预加载飞行姿态异常检测数据...")
        self.data_cache = {}
        
        for file_id in self.file_ids:
            file_path = Path(self.data_path) / f"{file_id}.csv"
            if not file_path.exists():
                continue
                
            df = pd.read_csv(file_path)
            self.data_cache[file_id] = df
    
    def _initialize_sequences(self):
        for file_id, df in self.data_cache.items():
            n_samples = len(df) - self.sequence_length + 1
            
            if n_samples > 0:
                for i in range(n_samples):
                    self.sequence_indices.append({
                        'file_id': file_id,
                        'start_idx': i,
                        'end_idx': i + self.sequence_length
                    })
                self.total_sequences += n_samples
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_info = self.sequence_indices[idx]
        df = self.data_cache[seq_info['file_id']]
        
        sequence_data = df.iloc[seq_info['start_idx']:seq_info['end_idx']]
        sequence = sequence_data[self.feature_columns].values
        
        if self.classification_mode:
            # 分类模式：返回序列的最后一个时间步的标签
            label = sequence_data['label'].iloc[-1]
            sequence = torch.FloatTensor(sequence)
            label = torch.LongTensor([label])
            return sequence, label
        else:
            # 回归模式：预测下一个时间步
            if seq_info['end_idx'] < len(df):
                target = df.iloc[seq_info['end_idx']]['roll']  # 以roll角为预测目标
                sequence = torch.FloatTensor(sequence)
                target = torch.FloatTensor([target])
                return sequence, target
            else:
                # 如果没有下一个时间步，返回当前时间步
                target = sequence_data['roll'].iloc[-1]
                sequence = torch.FloatTensor(sequence)
                target = torch.FloatTensor([target])
                return sequence, target

def get_flight_dataset(data_path: str, sequence_length: int = 50,
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      classification_mode: bool = True) -> Tuple[FlightAnomalyDataset, FlightAnomalyDataset, FlightAnomalyDataset]:
    """获取飞行姿态异常检测数据集"""
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"数据路径不存在: {data_path}")
    
    # 加载完整数据集
    file_path = data_path / "flight_attitude_anomaly_dataset.csv"
    if not file_path.exists():
        raise ValueError(f"飞行姿态数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    n_samples = len(df)
    
    # 按时间序列划分数据集
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 保存分割后的数据
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    train_df.to_csv(data_path / "flight_train.csv", index=False)
    val_df.to_csv(data_path / "flight_val.csv", index=False)
    test_df.to_csv(data_path / "flight_test.csv", index=False)
    
    # 创建数据集对象
    train_dataset = FlightAnomalyDataset(str(data_path), sequence_length, 
                                        file_ids=['flight_train'], 
                                        classification_mode=classification_mode)
    val_dataset = FlightAnomalyDataset(str(data_path), sequence_length, 
                                      file_ids=['flight_val'], 
                                      classification_mode=classification_mode)
    test_dataset = FlightAnomalyDataset(str(data_path), sequence_length, 
                                       file_ids=['flight_test'], 
                                       classification_mode=classification_mode)
    
    return train_dataset, val_dataset, test_dataset