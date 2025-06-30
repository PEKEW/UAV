"""
Data handling logic for the UAV health monitoring application
"""
import streamlit as st
import numpy as np
import pandas as pd
import h5py
import tempfile
import os
from pathlib import Path

class DataHandler:
    """Handles data loading, processing, and validation"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for data handling"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'data_type' not in st.session_state:
            st.session_state.data_type = 'unknown'
        if 'selected_time_range' not in st.session_state:
            st.session_state.selected_time_range = None
        if 'show_data_labels' not in st.session_state:
            st.session_state.show_data_labels = False
    
    def detect_data_type(self, filename):
        """Detect data type based on filename prefix"""
        filename_lower = filename.lower()
        if filename_lower.startswith('battery_'):
            return 'battery'
        elif filename_lower.startswith('flight_'):
            return 'flight'
        else:
            return 'unknown'
    
    def load_data(self, uploaded_file):
        """Load and process data file"""
        with st.spinner('加载数据...'):
            try:
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension not in ['.h5', '.hdf5']:
                    st.error(f"不支持的文件格式: {file_extension}，请使用H5格式")
                    return
                
                data_type = self.detect_data_type(uploaded_file.name)
                st.session_state.data_type = data_type
                
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    h5_data = self.load_h5_data(tmp_file_path)
                    if h5_data is None:
                        return
                    
                    st.success(f"成功读取H5数据，形状: {h5_data['data_shape']}")
                    
                    with st.expander("数据信息"):
                        st.write(f"**数据形状**: {h5_data['data_shape']}")
                        st.write(f"**特征数量**: {h5_data['n_features']}")
                        st.write(f"**样本数量**: {h5_data['n_samples']}")
                        st.write(f"**序列长度**: {h5_data['sequence_length']}")
                        st.write(f"**特征名称**: {', '.join(h5_data['feature_names'])}")
                        if 'label_stats' in h5_data:
                            st.write(f"**标签统计**: {h5_data['label_stats']}")
                    
                    if not self.validate_h5_data_type(h5_data, data_type):
                        return
                    
                    processed_data = self.process_h5_data_for_visualization(h5_data, data_type)
                    if processed_data is None:
                        return
                    
                    st.session_state.data = processed_data
                    st.session_state.data_loaded = True
                    st.session_state.model_detection_completed = False
                    st.session_state.selected_time_range = None
                        
                finally:
                    os.unlink(tmp_file_path)
                    
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
                st.session_state.data_loaded = False
                st.session_state.data = None
    
    def load_h5_data(self, h5_path):
        """Load H5 data file"""
        try:
            with h5py.File(h5_path, 'r') as f:
                data = f['data'][:]
                labels = f['labels'][:]
                
                if 'feature_names' in f:
                    feature_names_bytes = f['feature_names'][:]
                    feature_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                                   for name in feature_names_bytes]
                else:
                    feature_names = [f'feature_{i}' for i in range(data.shape[2])]
                
                data_type = f.attrs.get('data_type', 'unknown')
                sequence_length = f.attrs.get('sequence_length', 30)
                
                h5_data = {
                    'data': data,
                    'labels': labels,
                    'feature_names': feature_names,
                    'data_shape': data.shape,
                    'n_samples': data.shape[0],
                    'n_features': data.shape[2],
                    'sequence_length': sequence_length,
                    'data_type': data_type
                }
                
                if 'label_stats' in f.attrs:
                    h5_data['label_stats'] = f.attrs['label_stats']
                
                return h5_data
                
        except Exception as e:
            st.error(f"加载H5文件失败: {str(e)}")
            return None
    
    def validate_h5_data_type(self, h5_data, expected_data_type):
        """Validate H5 data type"""
        n_features = h5_data['n_features']
        feature_names = h5_data['feature_names']
        
        if expected_data_type == 'battery':
            expected_features = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                               'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
            if n_features != 7:
                st.error(f"电池数据应有7个特征，实际有{n_features}个")
                return False
        elif expected_data_type == 'flight':
            expected_features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                               'velocity', 'acceleration', 'altitude']
            if n_features != 9:
                st.error(f"飞行数据应有9个特征，实际有{n_features}个")
                return False
        else:
            st.error(f"未知数据类型: {expected_data_type}")
            return False
        
        if len(feature_names) == len(expected_features):
            mismatched = []
            for i, (expected, actual) in enumerate(zip(expected_features, feature_names)):
                if expected != actual:
                    mismatched.append(f"特征{i}: 期望'{expected}'，实际'{actual}'")
            
            if mismatched:
                st.warning("特征名称不完全匹配:")
                for msg in mismatched:
                    st.warning(f"  {msg}")
                st.info("将继续使用现有特征名称")
        
        if h5_data['sequence_length'] != 30:
            st.warning(f"序列长度为{h5_data['sequence_length']}，期望30")
        
        return True
    
    def process_h5_data_for_visualization(self, h5_data, data_type):
        """Process H5 data for visualization"""
        try:
            data = h5_data['data']
            labels = h5_data['labels']
            feature_names = h5_data['feature_names']
            
            n_samples, seq_len, n_features = data.shape
            
            time_data = np.arange(n_samples * seq_len) * 1.0
            
            features_dict = {}
            for i, feature_name in enumerate(feature_names):
                feature_data = data[:, :, i].flatten()
                features_dict[feature_name] = feature_data
            
            expanded_labels = np.repeat(labels, seq_len)
            data_labels = self.generate_h5_data_labels(labels, seq_len)
            
            processed_data = {
                'time': time_data,
                'features': features_dict,
                'original_samples': data,
                'original_labels': labels,
                'expanded_labels': expanded_labels,
                'data_labels': data_labels,
                'sample_info': {
                    'n_samples': n_samples,
                    'sequence_length': seq_len,
                    'n_features': n_features,
                    'feature_names': feature_names
                }
            }
            
            st.success(f"成功处理{data_type}数据: {n_samples}个样本，{len(time_data)}个时间点")
            st.info(f"时间范围: 0s - {time_data.max():.1f}s")
            st.info(f"可用特征: {', '.join(feature_names)}")
            
            return processed_data
            
        except Exception as e:
            st.error(f"H5数据处理失败: {str(e)}")
            return None
    
    def generate_h5_data_labels(self, sample_labels, seq_len):
        """Generate 30-second segment labels from H5 sample labels"""
        data_labels = []
        
        for i, label in enumerate(sample_labels):
            if label == 1:
                start_time = i * seq_len
                end_time = (i + 1) * seq_len
                data_labels.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'anomaly_ratio': 1.0,
                    'sample_id': i
                })
        
        return data_labels
    
    def process_battery_data(self, df):
        """Process battery data from CSV"""
        battery_features = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                           'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
        time_col = 'time_s' if 'time_s' in df.columns else 'time'
        
        if time_col not in df.columns:
            st.error("电池数据中未找到时间列 (time_s 或 time)")
            return None
        
        available_features = [col for col in battery_features if col in df.columns]
        if not available_features:
            st.error("电池数据中未找到预期特征列")
            return None
        
        processed_data = {
            'time': pd.to_numeric(df[time_col], errors='coerce').values,
            'features': {}
        }
        
        for feature in available_features:
            processed_data['features'][feature] = pd.to_numeric(df[feature], errors='coerce').values
        
        processed_data['data_labels'] = self.extract_data_labels(df)
        
        # Data cleaning
        valid_mask = ~np.isnan(processed_data['time'])
        for feature in available_features:
            valid_mask = valid_mask & ~np.isnan(processed_data['features'][feature])
        
        processed_data['time'] = processed_data['time'][valid_mask]
        for feature in available_features:
            processed_data['features'][feature] = processed_data['features'][feature][valid_mask]
        
        if len(processed_data['time']) == 0:
            st.error("处理后的电池数据为空")
            return None
        
        st.success(f"成功处理电池数据: {len(processed_data['time'])} 个数据点")
        st.info(f"时间范围: {processed_data['time'].min():.1f}s - {processed_data['time'].max():.1f}s")
        st.info(f"可用特征: {', '.join(available_features)}")
        
        return processed_data
    
    def process_flight_data(self, df):
        """Process flight data from CSV"""
        flight_features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
        time_col = 'time'
        
        if time_col not in df.columns:
            st.error("飞行数据中未找到时间列 (time)")
            return None
        
        available_features = [col for col in flight_features if col in df.columns]
        if not available_features:
            st.error("飞行数据中未找到预期特征列")
            return None
        
        processed_data = {
            'time': pd.to_numeric(df[time_col], errors='coerce').values,
            'features': {}
        }
        
        for feature in available_features:
            processed_data['features'][feature] = pd.to_numeric(df[feature], errors='coerce').values
        
        processed_data['data_labels'] = self.extract_data_labels(df)
        
        # Data cleaning
        valid_mask = ~np.isnan(processed_data['time'])
        for feature in available_features:
            valid_mask = valid_mask & ~np.isnan(processed_data['features'][feature])
        
        processed_data['time'] = processed_data['time'][valid_mask]
        for feature in available_features:
            processed_data['features'][feature] = processed_data['features'][feature][valid_mask]
        
        if len(processed_data['time']) == 0:
            st.error("处理后的飞行数据为空")
            return None
        
        st.success(f"成功处理飞行数据: {len(processed_data['time'])} 个数据点")
        st.info(f"时间范围: {processed_data['time'].min():.1f}s - {processed_data['time'].max():.1f}s")
        st.info(f"可用特征: {', '.join(available_features)}")
        
        return processed_data
    
    def extract_data_labels(self, df):
        """Extract anomaly labels from data, generate 30-second segment labels"""
        data_labels = []
        
        label_columns = ['label', 'anomaly', 'is_anomaly', 'sample_label']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            return data_labels
        
        try:
            time_col = 'time_s' if 'time_s' in df.columns else 'time'
            if time_col not in df.columns:
                return data_labels
            
            time_data = pd.to_numeric(df[time_col], errors='coerce').values
            labels = pd.to_numeric(df[label_col], errors='coerce').values
            
            valid_mask = ~(np.isnan(time_data) | np.isnan(labels))
            time_data = time_data[valid_mask]
            labels = labels[valid_mask]
            
            if len(time_data) == 0:
                return data_labels
            
            time_min, time_max = time_data.min(), time_data.max()
            current_time = time_min
            
            while current_time < time_max:
                segment_end = min(current_time + 30, time_max)
                
                segment_mask = (time_data >= current_time) & (time_data < segment_end)
                segment_labels = labels[segment_mask]
                
                if len(segment_labels) > 0:
                    anomaly_ratio = np.sum(segment_labels > 0) / len(segment_labels)
                    if anomaly_ratio > 0.1:
                        data_labels.append({
                            'start': current_time,
                            'end': segment_end,
                            'anomaly_ratio': anomaly_ratio
                        })
                
                current_time += 30
            
        except Exception as e:
            st.warning(f"提取异常标签失败: {str(e)}")
        
        return data_labels