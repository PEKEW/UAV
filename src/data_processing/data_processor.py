import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

class DataProcessor:
    """数据处理工具类，负责数据的加载、处理和转换"""
    
    @staticmethod
    def load_data(file):
        """加载数据文件"""
        try:
            file_extension = Path(file.name).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            if df.empty:
                raise ValueError("文件为空或无法读取")
            
            return df
        except Exception as e:
            raise Exception(f"数据加载失败: {str(e)}")
    
    @staticmethod
    def detect_columns(df):
        """检测数据列"""
        column_mapping = {}
        columns = df.columns.tolist()
        available_features = []
        
        # 检测时间列
        time_keywords = ['time_s', 'time', 'timestamp', '时间', 'Time', 'TIME', 't']
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in time_keywords):
                column_mapping['time'] = col
                break
        
        if 'time' not in column_mapping:
            column_mapping['time'] = 'index'
        
        # 检测特征列 - 支持电池和飞行数据
        feature_keywords = {
            # 电池相关特征
            'Ecell_V': (['ecell_v', 'voltage', 'volt', 'v', '电压', 'Ecell_V', 'cell_voltage'], '电压 (V)'),
            'I_mA': (['i_ma', 'current', 'amp', 'i', '电流', 'I_mA', 'I_A'], '电流 (mA)'),
            'Temperature__C': (['temperature', 'temp', '温度', 'Temperature__C', 'T'], '温度 (°C)'),
            'EnergyCharge_W_h': (['energycharge_w_h', 'charge_energy', 'EnergyCharge_W_h'], '充电能量 (Wh)'),
            'QCharge_mA_h': (['qcharge_ma_h', 'charge_capacity', 'QCharge_mA_h'], '充电容量 (mAh)'),
            'EnergyDischarge_W_h': (['energydischarge_w_h', 'discharge_energy', 'EnergyDischarge_W_h'], '放电能量 (Wh)'),
            'QDischarge_mA_h': (['qdischarge_ma_h', 'discharge_capacity', 'QDischarge_mA_h'], '放电容量 (mAh)'),
            # 飞行相关特征
            'x': (['x', 'position_x', 'pos_x'], 'X位置 (m)'),
            'y': (['y', 'position_y', 'pos_y'], 'Y位置 (m)'),
            'z': (['z', 'position_z', 'pos_z'], 'Z位置 (m)'),
            'roll': (['roll', 'φ'], '翻滚角 (度)'),
            'pitch': (['pitch', 'θ'], '俯仰角 (度)'),
            'yaw': (['yaw', 'ψ'], '偏航角 (度)'),
            'velocity': (['velocity', 'speed', 'v'], '速度 (m/s)'),
            'altitude': (['altitude', 'alt', 'height'], '高度 (m)')
        }
        
        for feature_type, (keywords, unit) in feature_keywords.items():
            for col in columns:
                if any(keyword.lower() in col.lower() for keyword in keywords):
                    available_features.append((feature_type.capitalize(), col, unit))
                    break
        
        # 添加其他数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if not any(col == feature[1] for feature in available_features):
                available_features.append((col, col, col))
        
        # 设置默认特征
        if available_features:
            column_mapping['true_values'] = available_features[0][1]
        
        return column_mapping, available_features
    
    @staticmethod
    def process_data(df, column_mapping):
        """处理数据"""
        try:
            processed_data = {}
            
            # 处理时间列
            if column_mapping['time'] == 'index':
                processed_data['time'] = np.arange(len(df))
            else:
                time_col = df[column_mapping['time']]
                processed_data['time'] = pd.to_numeric(time_col, errors='coerce').values
            
            # 处理位置数据
            position_cols = ['x', 'y', 'z']
            has_position = all(col in df.columns for col in position_cols)
            if has_position:
                for col in position_cols:
                    processed_data[col] = pd.to_numeric(df[col], errors='coerce').values
            
            # 处理姿态数据
            attitude_cols = ['roll', 'pitch', 'yaw']
            has_attitude = all(col in df.columns for col in attitude_cols)
            if has_attitude:
                for col in attitude_cols:
                    processed_data[col] = pd.to_numeric(df[col], errors='coerce').values
            
            # 处理速度、高度和加速度
            for col in ['velocity', 'altitude', 'acceleration']:
                if col in df.columns:
                    processed_data[col] = pd.to_numeric(df[col], errors='coerce').values
            
            # 处理电池相关特征
            battery_cols = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h', 
                           'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
            for col in battery_cols:
                if col in df.columns:
                    processed_data[col] = pd.to_numeric(df[col], errors='coerce').values
            
            # 处理标签
            if 'label' in df.columns:
                processed_data['label'] = pd.to_numeric(df['label'], errors='coerce').values
            
            # 处理主要特征值
            true_values = pd.to_numeric(df[column_mapping['true_values']], errors='coerce')
            if true_values.isna().all():
                raise ValueError(f"特征列 '{column_mapping['true_values']}' 无法转换为数值")
            processed_data['true_values'] = true_values.values
            
            # 标记数据类型
            processed_data['has_attitude'] = has_attitude
            processed_data['has_position'] = has_position
            
            # 处理缺失值
            valid_mask = ~(np.isnan(processed_data['time']) | np.isnan(processed_data['true_values']))
            for key in processed_data:
                if isinstance(processed_data[key], np.ndarray):
                    processed_data[key] = processed_data[key][valid_mask]
            
            if len(processed_data['time']) == 0:
                raise ValueError("处理后的数据为空")
            
            # 按时间排序
            sort_indices = np.argsort(processed_data['time'])
            for key in processed_data:
                if isinstance(processed_data[key], np.ndarray):
                    processed_data[key] = processed_data[key][sort_indices]
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"数据处理失败: {str(e)}")
    
    @staticmethod
    def calculate_velocity(time_data, x, y, z):
        """计算速度"""
        dt = np.diff(time_data)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        return np.append(velocity, velocity[-1])  # 补充最后一个点
    
    @staticmethod
    def normalize_features(features_array, model_type):
        """根据训练时的归一化参数对特征进行归一化"""
        try:
            # 根据模型类型使用对应的归一化参数
            if model_type == "flight":
                # 飞行模型使用MinMaxScaler，范围[0,1]
                # 使用训练时的归一化参数
                feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
                
                # 创建MinMaxScaler并使用训练时的参数范围
                # 这些参数应该与create_flight_attitude_dataset.py中的参数一致
                scaler = MinMaxScaler()
                
                # 使用训练数据的实际统计信息进行归一化
                # 这些范围来自flight_attitude_anomaly_dataset.csv的实际数据分析
                feature_ranges = {
                    'x': (-83.864, 90.049),           # 实际范围: [-83.864, 90.049]
                    'y': (-86.240, 89.167),           # 实际范围: [-86.240, 89.167] 
                    'z': (-45.759, 200.162),          # 实际范围: [-45.759, 200.162]
                    'roll': (-63.354, 64.227),        # 实际范围: [-63.354, 64.227]
                    'pitch': (-33.331, 28.090),       # 实际范围: [-33.331, 28.090]
                    'yaw': (-7.911, 100.604),         # 实际范围: [-7.911, 100.604]
                    'velocity': (-0.643, 43.322),     # 实际范围: [-0.643, 43.322]
                    'acceleration': (-7.653, 10.220), # 实际范围: [-7.653, 10.220]
                    'altitude': (54.241, 300.162)     # 实际范围: [54.241, 300.162]
                }
                
                # 手动归一化到[0,1]范围
                normalized_features = features_array.copy()
                for i, feature in enumerate(feature_columns):
                    min_val, max_val = feature_ranges[feature]
                    normalized_features[:, i] = (features_array[:, i] - min_val) / (max_val - min_val)
                    # 确保在[0,1]范围内
                    normalized_features[:, i] = np.clip(normalized_features[:, i], 0, 1)
                
                return normalized_features
                
            elif model_type == "battery":
                # 电池模型不使用归一化（与训练脚本一致）
                return features_array
            else:
                return features_array
                
        except Exception as e:
            print(f"特征归一化失败: {str(e)}")
            return features_array 