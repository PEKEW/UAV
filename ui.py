import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
from pathlib import Path
import h5py
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="UAV健康-DEMO", 
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 48px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .control-panel {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .status-online {
        background-color: #f44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-offline {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .control-panel{
        font-size: 28px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        text-align: left;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

class BatteryAnalysisApp:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        if 'current_status' not in st.session_state:
            st.session_state.current_status = 'offline'
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        if 'model_detection_completed' not in st.session_state:
            st.session_state.model_detection_completed = False
        if 'data_type' not in st.session_state:
            st.session_state.data_type = 'unknown'
        if 'selected_time_range' not in st.session_state:
            st.session_state.selected_time_range = None
        if 'show_data_labels' not in st.session_state:
            st.session_state.show_data_labels = False

    def create_header(self):
        st.markdown('<h1 class="main-header">电池健康管理-DEMO</h1>', unsafe_allow_html=True)
    
    def create_control_panel(self):
        with st.container():
            st.markdown('<div class="control-panel">控制面板</div>', unsafe_allow_html=True)
            
            st.markdown("### 监测模式 ")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("实时监测", use_container_width=True):
                    st.toast("️不行！", icon="⚠️")
                    st.session_state.current_status = 'offline'  # 保持离线状态
            with col2:
                if st.button("离线监测", use_container_width=True):
                    st.session_state.current_status = 'offline'
            
            if st.session_state.current_status == 'online':
                st.markdown('<div class="status-online"> 没有实时监测！</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-offline">离线监测</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 数据选择")
            uploaded_file = st.file_uploader(
                "点击导入或拖拽数据文件到此处",
                type=['h5', 'hdf5'],
                help="支持H5格式（battery_开头为电池数据，flight_开头为飞行数据）"
            )
            
            if st.button("导入数据", use_container_width=True, disabled=not (uploaded_file is not None and st.session_state.current_status == 'offline')):
                    self.load_data(uploaded_file)
            
            if st.session_state.data_loaded:
                st.success("数据已导入")
                if hasattr(st.session_state, 'data_type'):
                    st.info(f"数据类型: {st.session_state.data_type}")
            else:
                st.info("未导入数据")
            
            st.markdown("---")
            
            st.markdown("### 异常识别")
            st.markdown("#### 模型选择")
            uploaded_model = st.file_uploader(
                "点击导入或拖拽模型文件到此处",
                type=['pkl', 'joblib', 'h5', 'pth', 'pt', 'onnx'],
                help="支持pkl, joblib, h5, pth, pt, onnx格式",
                key="model_uploader"
            )
            
            if st.button("加载模型", use_container_width=True, disabled=not (uploaded_model is not None and st.session_state.current_status == 'offline')):
                self.load_model(uploaded_model)
            
            if st.session_state.model_loaded:
                st.success("模型已加载")
                if st.session_state.model_info:
                    with st.expander("模型信息"):
                        for key, value in st.session_state.model_info.items():
                            st.write(f"**{key}**: {value}")
            else:
                st.info("未加载模型")
            
            if st.button("开始识别(务必确保模型类型和识别特征匹配)", use_container_width=True, disabled=not (st.session_state.data_loaded and st.session_state.model_loaded)):
                # 检查模型和数据兼容性
                if st.session_state.model_loaded and st.session_state.data_loaded:
                    model_type = st.session_state.model.get('model_type', 'unknown')
                    data_type = st.session_state.data_type
                    
                    is_compatible, compatibility_msg = self.validate_model_data_compatibility(model_type, data_type)
                    
                    if not is_compatible:
                        st.error(f"模型与数据不兼容: {compatibility_msg}")
                        st.error("请确保:")
                        st.error("1. 电池模型文件名以 'battery_' 开头，用于电池数据")
                        st.error("2. 飞行模型文件名以 'flight_' 开头，用于飞行数据")
                        return
                
                with st.spinner('正在执行模型识别...'):
                    if hasattr(st.session_state, 'data'):
                        anomaly_regions = self.generate_anomaly_detection(st.session_state.data['time'])
                        st.session_state.data['anomaly_regions'] = anomaly_regions
                        st.session_state.model_detection_completed = True
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_chart_area(self):
        st.markdown("##  数据可视化与异常检测")
        if st.session_state.data_loaded and st.session_state.data is not None:
            data = st.session_state.data
            data_type = getattr(st.session_state, 'data_type', 'unknown')
            total_points = len(data['time'])
            time_min, time_max = data['time'].min(), data['time'].max()
            total_duration = time_max - time_min
            st.info(f"数据总量: {total_points} 个数据点 | 时间范围: {time_min:.1f}s - {time_max:.1f}s | 总时长: {total_duration:.1f}s")
            self.create_anomaly_label_control()
            self.create_time_range_selector(time_min, time_max)
            filtered_data = self.filter_data_by_time_range(data)
            if data_type == 'battery':
                self.create_battery_visualization(filtered_data)
            elif data_type == 'flight':
                self.create_flight_visualization(filtered_data)
            else:
                st.warning("未知数据类型，无法创建可视化")
        
        else:
            st.info("请先上传数据文件开始分析")
    
    def create_anomaly_label_control(self):
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_labels = st.checkbox(
                "显示数据中的异常标签",
                value=st.session_state.show_data_labels,
                help="显示数据中原有的异常标签（30秒为单位）"
            )
            
            if show_labels != st.session_state.show_data_labels:
                st.session_state.show_data_labels = show_labels
                st.rerun()
        
        with col2:
            if st.session_state.show_data_labels and st.session_state.data_loaded:
                # 检查数据中是否存在标签
                data = st.session_state.data
                has_labels = self.check_data_has_labels(data)
                if has_labels:
                    st.success("✓ 数据中存在异常标签")
                else:
                    st.warning("⚠ 数据中未找到异常标签")
        
        st.markdown("---")
    
    def check_data_has_labels(self, data):
        """检查数据中是否存在异常标签"""
        return 'data_labels' in data and len(data['data_labels']) > 0
    
    def create_time_range_selector(self, time_min, time_max):
        """创建时间范围选择器"""
        st.markdown("### 时间范围选择")
        
        total_duration = time_max - time_min
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 计算30秒为单位的范围
            start_options = np.arange(time_min, time_max, 30)
            start_time = st.selectbox(
                "起始时间 (秒)",
                options=start_options,
                format_func=lambda x: f"{x:.0f}s",
                key="start_time_selector"
            )
        
        with col2:
            # 结束时间选项，从起始时间开始，以30秒为单位
            end_options = np.arange(start_time + 30, time_max + 30, 30)
            end_options = end_options[end_options <= time_max]
            if len(end_options) == 0:
                end_options = [time_max]
            
            end_time = st.selectbox(
                "结束时间 (秒)",
                options=end_options,
                format_func=lambda x: f"{x:.0f}s",
                key="end_time_selector"
            )
        
        with col3:
            if st.button("应用时间范围"):
                st.session_state.selected_time_range = (start_time, end_time)
                st.success(f"已选择时间范围: {start_time:.0f}s - {end_time:.0f}s")
                st.rerun()
        
        with col4:
            if st.button("重置范围"):
                st.session_state.selected_time_range = None
                st.success("已重置为显示全部数据")
                st.rerun()
        
        # 显示当前选择的范围
        if st.session_state.selected_time_range:
            start, end = st.session_state.selected_time_range
            duration = end - start
            st.info(f"当前显示范围: {start:.0f}s - {end:.0f}s (时长: {duration:.0f}s)")
        else:
            st.info(f"当前显示全部数据 (总时长: {total_duration:.0f}s)")
        
        st.markdown("---")
    
    def filter_data_by_time_range(self, data):
        """根据选择的时间范围过滤数据"""
        if st.session_state.selected_time_range is None:
            return data
        
        start_time, end_time = st.session_state.selected_time_range
        
        # 创建时间掩码
        time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
        
        # 过滤数据
        filtered_data = {
            'time': data['time'][time_mask],
            'features': {}
        }
        
        # 过滤所有特征
        for feature_name, feature_data in data['features'].items():
            filtered_data['features'][feature_name] = feature_data[time_mask]
        
        # 过滤异常区域（如果存在）
        if 'anomaly_regions' in data:
            filtered_anomaly_regions = []
            for region in data['anomaly_regions']:
                region_start, region_end = region['start'], region['end']
                # 只保留与选择时间范围有重叠的异常区域
                if region_start <= end_time and region_end >= start_time:
                    # 调整异常区域边界
                    adjusted_region = region.copy()
                    adjusted_region['start'] = max(region_start, start_time)
                    adjusted_region['end'] = min(region_end, end_time)
                    filtered_anomaly_regions.append(adjusted_region)
            filtered_data['anomaly_regions'] = filtered_anomaly_regions
        
        # 过滤数据标签（如果存在）
        if 'data_labels' in data:
            filtered_data_labels = []
            for label in data['data_labels']:
                label_start, label_end = label['start'], label['end']
                # 只保留与选择时间范围有重叠的数据标签
                if label_start <= end_time and label_end >= start_time:
                    # 调整标签边界
                    adjusted_label = label.copy()
                    adjusted_label['start'] = max(label_start, start_time)
                    adjusted_label['end'] = min(label_end, end_time)
                    filtered_data_labels.append(adjusted_label)
            filtered_data['data_labels'] = filtered_data_labels
        
        return filtered_data
    
    def detect_data_type(self, filename):
        """根据文件名前缀检测数据类型"""
        filename_lower = filename.lower()
        if filename_lower.startswith('battery_'):
            return 'battery'
        elif filename_lower.startswith('flight_'):
            return 'flight'
        else:
            return 'unknown'

    def load_data(self, uploaded_file):
        with st.spinner('加载数据...'):
            try:
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension not in ['.h5', '.hdf5']:
                    st.error(f"不支持的文件格式: {file_extension}，请使用H5格式")
                    return
                
                data_type = self.detect_data_type(uploaded_file.name)
                st.session_state.data_type = data_type
                
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 加载H5数据
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
                    
                    # 验证数据类型兼容性
                    if not self.validate_h5_data_type(h5_data, data_type):
                        return
                    
                    # 处理H5数据用于可视化
                    processed_data = self.process_h5_data_for_visualization(h5_data, data_type)
                    if processed_data is None:
                        return
                    
                    st.session_state.data = processed_data
                    st.session_state.data_loaded = True
                    st.session_state.model_detection_completed = False
                    st.session_state.selected_time_range = None  # 重置时间选择
                        
                finally:
                    # 清理临时文件
                    os.unlink(tmp_file_path)
                    
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
                st.session_state.data_loaded = False
                st.session_state.data = None

    def load_h5_data(self, h5_path):
        """加载H5数据文件"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # 读取基本数据
                data = f['data'][:]  # shape: (n_samples, 30, n_features)
                labels = f['labels'][:]  # shape: (n_samples,)
                
                # 读取特征名称
                if 'feature_names' in f:
                    feature_names_bytes = f['feature_names'][:]
                    feature_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                                   for name in feature_names_bytes]
                else:
                    feature_names = [f'feature_{i}' for i in range(data.shape[2])]
                
                # 读取元数据
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
                
                # 如果有标签统计信息
                if 'label_stats' in f.attrs:
                    h5_data['label_stats'] = f.attrs['label_stats']
                
                return h5_data
                
        except Exception as e:
            st.error(f"加载H5文件失败: {str(e)}")
            return None
    
    def validate_h5_data_type(self, h5_data, expected_data_type):
        """验证H5数据类型"""
        # 验证特征数量
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
        
        # 验证特征名称（如果可用）
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
        
        # 验证序列长度
        if h5_data['sequence_length'] != 30:
            st.warning(f"序列长度为{h5_data['sequence_length']}，期望30")
        
        return True
    
    def process_h5_data_for_visualization(self, h5_data, data_type):
        """处理H5数据用于可视化"""
        try:
            data = h5_data['data']  # (n_samples, 30, n_features)
            labels = h5_data['labels']  # (n_samples,)
            feature_names = h5_data['feature_names']
            
            # 将3D数据展开为时间序列以便可视化
            # 假设每个样本代表30秒，样本间连续
            n_samples, seq_len, n_features = data.shape
            
            # 创建时间轴：每个样本30秒，样本间连续
            time_data = np.arange(n_samples * seq_len) * 1.0  # 假设每秒一个数据点
            
            # 展开特征数据
            features_dict = {}
            for i, feature_name in enumerate(feature_names):
                # 将所有样本的这个特征连接成一个长时间序列
                feature_data = data[:, :, i].flatten()  # (n_samples * 30,)
                features_dict[feature_name] = feature_data
            
            # 展开标签，每个样本的30个时间点都使用同一个标签
            expanded_labels = np.repeat(labels, seq_len)
            
            # 生成数据标签（基于原始标签）
            data_labels = self.generate_h5_data_labels(labels, seq_len)
            
            processed_data = {
                'time': time_data,
                'features': features_dict,
                'original_samples': data,  # 保留原始3D数据用于模型推理
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
        """根据H5样本标签生成30秒段的标签"""
        data_labels = []
        
        for i, label in enumerate(sample_labels):
            if label == 1:  # 异常样本
                start_time = i * seq_len
                end_time = (i + 1) * seq_len
                data_labels.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'anomaly_ratio': 1.0,  # H5中整个样本都是异常
                    'sample_id': i
                })
        
        return data_labels

    def show_detection_statistics(self, anomaly_regions):
        """显示模型检测统计信息"""
        if not st.session_state.data_loaded or not anomaly_regions:
            return
        
        data = st.session_state.data
        
        with st.expander("🔍 检测结果分析", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("模型检测异常区域", f"{len(anomaly_regions)} 个")
                
                # 计算平均置信度
                confidences = [region.get('confidence', 0.5) for region in anomaly_regions]
                avg_confidence = np.mean(confidences) if confidences else 0
                st.metric("平均置信度", f"{avg_confidence:.2f}")
            
            with col2:
                # 数据标签统计
                if 'data_labels' in data and data['data_labels']:
                    st.metric("数据标签异常区域", f"{len(data['data_labels'])} 个")
                    
                    # 计算重叠率
                    overlap_count = self.calculate_overlap_rate(anomaly_regions, data['data_labels'])
                    st.metric("区域重叠数", f"{overlap_count} 个")
                else:
                    st.metric("数据标签异常区域", "0 个")
                    st.metric("区域重叠数", "N/A")
            
            with col3:
                # 总体检测覆盖率
                total_time = data['time'].max() - data['time'].min()
                anomaly_time = sum(region['end'] - region['start'] for region in anomaly_regions)
                coverage_rate = anomaly_time / total_time if total_time > 0 else 0
                st.metric("异常时间覆盖率", f"{coverage_rate:.1%}")
                
                # 置信度分布
                if confidences:
                    high_conf_count = sum(1 for c in confidences if c > 0.8)
                    st.metric("高置信度区域 (>0.8)", f"{high_conf_count} 个")
    
    def calculate_overlap_rate(self, model_regions, data_labels):
        """计算模型检测与数据标签的重叠率"""
        overlap_count = 0
        
        for model_region in model_regions:
            model_start, model_end = model_region['start'], model_region['end']
            
            for data_label in data_labels:
                label_start, label_end = data_label['start'], data_label['end']
                
                # 检查是否有重叠
                if not (model_end <= label_start or model_start >= label_end):
                    overlap_count += 1
                    break  # 一个模型区域只计算一次重叠
        
        return overlap_count

    def process_data_by_type(self, df, data_type):
        """根据数据类型处理数据"""
        try:
            if data_type == 'battery':
                return self.process_battery_data(df)
            elif data_type == 'flight':
                return self.process_flight_data(df)
            else:
                st.error("未知数据类型，请确保文件名以 'battery_' 或 'flight_' 开头")
                return None
        except Exception as e:
            st.error(f"数据处理失败: {str(e)}")
            return None
    
    def process_battery_data(self, df):
        """处理电池数据"""
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
        
        # 检测异常标签
        processed_data['data_labels'] = self.extract_data_labels(df)
        
        # 数据清理
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
        """处理飞行数据"""
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
        
        # 检测异常标签
        processed_data['data_labels'] = self.extract_data_labels(df)
        
        # 数据清理
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
        """提取数据中的异常标签，生成30秒段的标签"""
        data_labels = []
        
        # 检查是否有标签列
        label_columns = ['label', 'anomaly', 'is_anomaly', 'sample_label']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            return data_labels
        
        try:
            # 获取时间列
            time_col = 'time_s' if 'time_s' in df.columns else 'time'
            if time_col not in df.columns:
                return data_labels
            
            time_data = pd.to_numeric(df[time_col], errors='coerce').values
            labels = pd.to_numeric(df[label_col], errors='coerce').values
            
            # 过滤有效数据
            valid_mask = ~(np.isnan(time_data) | np.isnan(labels))
            time_data = time_data[valid_mask]
            labels = labels[valid_mask]
            
            if len(time_data) == 0:
                return data_labels
            
            # 按30秒段分组，查找异常段
            time_min, time_max = time_data.min(), time_data.max()
            current_time = time_min
            
            while current_time < time_max:
                segment_end = min(current_time + 30, time_max)
                
                # 找到这个时间段内的数据
                segment_mask = (time_data >= current_time) & (time_data < segment_end)
                segment_labels = labels[segment_mask]
                
                if len(segment_labels) > 0:
                    # 如果这个30秒段内有超过一定比例的异常标签，就标记为异常段
                    anomaly_ratio = np.sum(segment_labels > 0) / len(segment_labels)
                    if anomaly_ratio > 0.1:  # 10%以上的点为异常就认为是异常段
                        data_labels.append({
                            'start': current_time,
                            'end': segment_end,
                            'anomaly_ratio': anomaly_ratio
                        })
                
                current_time += 30
            
        except Exception as e:
            st.warning(f"提取异常标签失败: {str(e)}")
        
        return data_labels

    def create_battery_visualization(self, data):
        """创建电池数据可视化"""
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('电池电压 (Ecell_V)', '电流 (I_mA)', '充电能量 (EnergyCharge_W_h)', 
                           '放电能量 (EnergyDischarge_W_h)', '温度 (Temperature__C)', '容量变化'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            vertical_spacing=0.08
        )
        
        time_data = data['time']
        features = data['features']
        
        # 定义颜色
        colors = {
            'Ecell_V': 'blue',
            'I_mA': 'red', 
            'EnergyCharge_W_h': 'red',
            'EnergyDischarge_W_h': 'orange',
            'QCharge_mA_h': 'purple',
            'QDischarge_mA_h': 'brown',
            'Temperature__C': 'magenta'
        }
        
        # 1. 电池电压
        if 'Ecell_V' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['Ecell_V'],
                mode='lines',
                name='电池电压',
                line=dict(color=colors['Ecell_V'], width=1)
            ), row=1, col=1)
        
        # 2. 电流
        if 'I_mA' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['I_mA'],
                mode='lines',
                name='电流',
                line=dict(color=colors['I_mA'], width=1)
            ), row=1, col=2)
        
        # 3. 充电能量
        if 'EnergyCharge_W_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['EnergyCharge_W_h'],
                mode='lines',
                name='充电能量',
                line=dict(color=colors['EnergyCharge_W_h'], width=1)
            ), row=2, col=1)
        
        # 4. 放电能量
        if 'EnergyDischarge_W_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['EnergyDischarge_W_h'],
                mode='lines',
                name='放电能量',
                line=dict(color=colors['EnergyDischarge_W_h'], width=1)
            ), row=2, col=2)
        
        # 5. 温度
        if 'Temperature__C' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['Temperature__C'],
                mode='lines',
                name='温度',
                line=dict(color=colors['Temperature__C'], width=1)
            ), row=3, col=1)
        
        # 6. 容量变化（充电和放电容量在同一图中）
        if 'QCharge_mA_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['QCharge_mA_h'],
                mode='lines',
                name='充电容量',
                line=dict(color=colors['QCharge_mA_h'], width=1)
            ), row=3, col=2)
        if 'QDischarge_mA_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['QDischarge_mA_h'],
                mode='lines',
                name='放电容量',
                line=dict(color=colors['QDischarge_mA_h'], width=1)
            ), row=3, col=2)
        
        # 添加模型检测的异常区域（红色，更透明）
        if st.session_state.model_detection_completed and 'anomaly_regions' in data:
            for i, region in enumerate(data['anomaly_regions']):
                start, end = region['start'], region['end']
                confidence = region.get('confidence', 0.5)
                # 为每个子图添加异常区域
                for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
                    # 获取该子图的数据范围
                    if row == 1 and col == 1 and 'Ecell_V' in features:
                        y_data = features['Ecell_V']
                    elif row == 1 and col == 2 and 'I_mA' in features:
                        y_data = features['I_mA']
                    elif row == 2 and col == 1 and 'EnergyCharge_W_h' in features:
                        y_data = features['EnergyCharge_W_h']
                    elif row == 2 and col == 2 and 'EnergyDischarge_W_h' in features:
                        y_data = features['EnergyDischarge_W_h']
                    elif row == 3 and col == 1 and 'Temperature__C' in features:
                        y_data = features['Temperature__C']
                    elif row == 3 and col == 2 and 'QCharge_mA_h' in features:
                        y_data = features['QCharge_mA_h']
                    else:
                        continue
                    
                    y_min, y_max = y_data.min(), y_data.max()
                    y_range = y_max - y_min
                    y_min = y_min - y_range * 0.1
                    y_max = y_max + y_range * 0.1
                    
                    # 使用红色表示模型检测的异常，更明显的样式
                    alpha = 0.4 + 0.4 * confidence  # 置信度越高，颜色越深，基础透明度提高
                    fig.add_trace(go.Scatter(
                        x=[start, start, end, end, start],
                        y=[y_min, y_max, y_max, y_min, y_min],
                        fill="toself",
                        fillcolor=f"rgba(255, 20, 20, {alpha})",  # 更鲜艳的红色
                        line=dict(width=0),  # 去掉边框
                        mode="none",  # 去掉顶点
                        showlegend=True if i == 0 else False,
                        name="模型检测异常" if i == 0 else None,
                        hovertemplate=f"模型检测异常<br>时间: {start:.1f}s-{end:.1f}s<br>置信度: {confidence:.3f}<extra></extra>"
                    ), row=row, col=col)
        
        # 添加数据中的异常标签（橙色/黄色）
        if st.session_state.show_data_labels and 'data_labels' in data:
            for j, label in enumerate(data['data_labels']):
                start, end = label['start'], label['end']
                anomaly_ratio = label.get('anomaly_ratio', 0.5)
                # 为每个子图添加数据标签
                for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
                    # 获取该子图的数据范围
                    if row == 1 and col == 1 and 'Ecell_V' in features:
                        y_data = features['Ecell_V']
                    elif row == 1 and col == 2 and 'I_mA' in features:
                        y_data = features['I_mA']
                    elif row == 2 and col == 1 and 'EnergyCharge_W_h' in features:
                        y_data = features['EnergyCharge_W_h']
                    elif row == 2 and col == 2 and 'EnergyDischarge_W_h' in features:
                        y_data = features['EnergyDischarge_W_h']
                    elif row == 3 and col == 1 and 'Temperature__C' in features:
                        y_data = features['Temperature__C']
                    elif row == 3 and col == 2 and 'QCharge_mA_h' in features:
                        y_data = features['QCharge_mA_h']
                    else:
                        continue
                    
                    y_min, y_max = y_data.min(), y_data.max()
                    y_range = y_max - y_min
                    y_min = y_min - y_range * 0.1
                    y_max = y_max + y_range * 0.1
                    
                    # 使用橙色表示数据标签中的异常
                    fig.add_trace(go.Scatter(
                        x=[start, start, end, end, start],
                        y=[y_min, y_max, y_max, y_min, y_min],
                        fill="toself",
                        fillcolor="rgba(255, 165, 0, 0.25)",  # 橙色，轻微降低透明度
                        line=dict(width=0),  # 去掉边框
                        mode="none",  # 去掉顶点
                        showlegend=True if j == 0 else False,
                        name="数据标签异常" if j == 0 else None,
                        hovertemplate=f"数据标签异常<br>时间: {start:.1f}s-{end:.1f}s<br>异常比例: {anomaly_ratio:.3f}<extra></extra>"
                    ), row=row, col=col)
        
        fig.update_layout(
            title='电池数据异常检测',
            height=1000,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key="battery_chart")
        
        # 显示统计信息
        self.show_battery_metrics(data)
    
    def create_flight_visualization(self, data):
        """创建飞行数据可视化"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('姿态角 (Roll, Pitch, Yaw)', '速度', '3D轨迹', '位置 (X, Y, Z)'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter3d", "secondary_y": False}, {"type": "scatter"}]],
            vertical_spacing=0.12
        )
        
        time_data = data['time']
        features = data['features']
        
        # 跟踪哪些子图有数据
        subplot_has_data = {
            (1, 1): False,  # 姿态角
            (1, 2): False,  # 速度
            (2, 1): False,  # 3D轨迹
            (2, 2): False   # 位置
        }
        
        # 1. 姿态角
        attitude_colors = {'roll': 'red', 'pitch': 'blue', 'yaw': 'orange'}
        for attitude in ['roll', 'pitch', 'yaw']:
            if attitude in features:
                fig.add_trace(go.Scatter(
                    x=time_data, y=features[attitude],
                    mode='lines',
                    name=attitude.capitalize(),
                    line=dict(color=attitude_colors[attitude], width=1)
                ), row=1, col=1)
                subplot_has_data[(1, 1)] = True
        
        # 2. 速度
        if 'velocity' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['velocity'],
                mode='lines',
                name='速度',
                line=dict(color='purple', width=1)
            ), row=1, col=2)
            subplot_has_data[(1, 2)] = True
        
        # 3. 3D轨迹 - H5样本分段显示，避免不连续样本间的连线
        if 'x' in features and 'y' in features and 'z' in features:
            # 检查是否为H5数据（样本式）
            if 'original_samples' in data:
                # H5数据：每个样本单独绘制，避免样本间连线
                original_samples = data['original_samples']  # (n_samples, 30, n_features)
                sample_info = data['sample_info']
                n_samples = sample_info['n_samples']
                feature_names = sample_info['feature_names']
                
                # 获取x, y, z特征的索引
                x_idx = feature_names.index('x') if 'x' in feature_names else None
                y_idx = feature_names.index('y') if 'y' in feature_names else None  
                z_idx = feature_names.index('z') if 'z' in feature_names else None
                
                if x_idx is not None and y_idx is not None and z_idx is not None:
                    # 为每个样本创建独立的轨迹段
                    for i in range(n_samples):
                        sample_x = original_samples[i, :, x_idx]
                        sample_y = original_samples[i, :, y_idx]
                        sample_z = original_samples[i, :, z_idx]
                        sample_time = np.arange(i * 30, (i + 1) * 30)
                        
                        # 根据是否为异常样本选择颜色
                        is_anomaly = False
                        if 'original_labels' in data and i < len(data['original_labels']):
                            is_anomaly = data['original_labels'][i] == 1
                        
                        color = 'rgba(255, 100, 100, 0.8)' if is_anomaly else 'rgba(70, 130, 180, 0.8)'
                        
                        fig.add_trace(go.Scatter3d(
                            x=sample_x,
                            y=sample_y, 
                            z=sample_z,
                            mode='markers',
                            marker=dict(size=1, color=color),
                            name='3D轨迹' if i == 0 else None,
                            showlegend=(i == 0),  # 只显示一个3D轨迹图例
                            hovertemplate=f'样本{i+1}<br>时间: %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
                            text=[f'{t:.1f}s' for t in sample_time]
                        ), row=2, col=1)
            else:
                # CSV数据：连续轨迹
                fig.add_trace(go.Scatter3d(
                    x=features['x'], 
                    y=features['y'], 
                    z=features['z'],
                    mode='lines',
                    line=dict(color='rgba(70, 130, 180, 0.8)', width=4),
                    name='3D飞行轨迹',
                    showlegend=True,
                    hovertemplate='时间: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    text=[f'{t:.1f}s' for t in time_data]
                ), row=2, col=1)
            
            subplot_has_data[(2, 1)] = True
        
                # 4. X, Y, Z 位置时间序列
        pos_colors = {'x': 'red', 'y': 'red', 'z': 'blue'}
        for pos in ['x', 'y', 'z']:
            if pos in features:
                fig.add_trace(go.Scatter(
                    x=time_data, y=features[pos],
                    mode='lines',
                    name=f'{pos.upper()}位置',
                    line=dict(color=pos_colors[pos], width=1)
                ), row=2, col=2)
                subplot_has_data[(2, 2)] = True
                
        
        # 添加模型检测的异常区域（红色）
        if st.session_state.model_detection_completed and 'anomaly_regions' in data:
            for k, region in enumerate(data['anomaly_regions']):
                start, end = region['start'], region['end']
                confidence = region.get('confidence', 0.5)
                # 只为有数据的2D时间序列图添加异常区域背景
                for row, col in [(1, 1), (1, 2), (2, 2)]:  # 排除3D图 (2, 1)
                    if subplot_has_data[(row, col)]:  # 只对有数据的子图添加
                        # 获取该子图的数据范围
                        if row == 1 and col == 1:  # 姿态角
                            y_data = np.concatenate([features[att] for att in ['roll', 'pitch', 'yaw'] if att in features])
                        elif row == 1 and col == 2 and 'velocity' in features:  # 速度
                            y_data = features['velocity']
                        elif row == 2 and col == 2:  # 位置
                            y_data = np.concatenate([features[pos] for pos in ['x', 'y', 'z'] if pos in features])
                        else:
                            continue
                        
                        y_min, y_max = y_data.min(), y_data.max()
                        y_range = y_max - y_min
                        y_min = y_min - y_range * 0.1
                        y_max = y_max + y_range * 0.1
                        
                        alpha = 0.4 + 0.4 * confidence 
                        fig.add_trace(go.Scatter(
                            x=[start, start, end, end, start],
                            y=[y_min, y_max, y_max, y_min, y_min],
                            fill="toself",
                            fillcolor=f"rgba(255, 20, 20, {alpha})", 
                            line=dict(width=0),
                            mode="none", 
                            showlegend=True if k == 0 else False,
                            name="模型检测异常" if k == 0 else None,
                            hovertemplate=f"模型检测异常<br>时间: {start:.1f}s-{end:.1f}s<br>置信度: {confidence:.3f}<extra></extra>"
                        ), row=row, col=col)
        
        # 添加数据中的异常标签（橙色）
        if st.session_state.show_data_labels and 'data_labels' in data:
            for m, label in enumerate(data['data_labels']):
                start, end = label['start'], label['end']
                anomaly_ratio = label.get('anomaly_ratio', 0.5)
                # 只为有数据的2D时间序列图添加异常区域背景
                for row, col in [(1, 1), (1, 2), (2, 2)]:  # 排除3D图 (2, 1)
                    if subplot_has_data[(row, col)]:  # 只对有数据的子图添加
                        # 获取该子图的数据范围
                        if row == 1 and col == 1:  # 姿态角
                            y_data = np.concatenate([features[att] for att in ['roll', 'pitch', 'yaw'] if att in features])
                        elif row == 1 and col == 2 and 'velocity' in features:  # 速度
                            y_data = features['velocity']
                        elif row == 2 and col == 2:  # 位置
                            y_data = np.concatenate([features[pos] for pos in ['x', 'y', 'z'] if pos in features])
                        else:
                            continue
                        
                        y_min, y_max = y_data.min(), y_data.max()
                        y_range = y_max - y_min
                        y_min = y_min - y_range * 0.1
                        y_max = y_max + y_range * 0.1
                        
                        # 使用橙色表示数据标签中的异常
                        fig.add_trace(go.Scatter(
                            x=[start, start, end, end, start],
                            y=[y_min, y_max, y_max, y_min, y_min],
                            fill="toself",
                            fillcolor="rgba(255, 165, 0, 0.25)",  # 橙色，轻微降低透明度
                            line=dict(width=0),  # 去掉边框
                            mode="none",  # 去掉顶点
                            showlegend=True if m == 0 else False,
                            name="数据标签异常" if m == 0 else None,
                            hovertemplate=f"数据标签异常<br>时间: {start:.1f}s-{end:.1f}s<br>异常比例: {anomaly_ratio:.3f}<extra></extra>"
                        ), row=row, col=col)
        
        fig.update_layout(
            title='飞行数据异常检测',
            height=800,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key="flight_chart")
        
        # 显示统计信息
        self.show_flight_metrics(data)
    
    def show_battery_metrics(self, data):
        """显示电池数据统计信息"""
        st.markdown("### 电池数据统计信息")
        
        features = data['features']
        cols = st.columns(min(len(features), 4))
        
        for i, (feature_name, feature_data) in enumerate(features.items()):
            with cols[i % 4]:
                st.metric(
                    label=feature_name,
                    value=f"{feature_data.mean():.3f}",
                    delta=f"范围: {feature_data.min():.3f} - {feature_data.max():.3f}"
                )
    
    def show_flight_metrics(self, data):
        """显示飞行数据统计信息"""
        st.markdown("### 飞行数据统计信息")
        
        features = data['features']
        cols = st.columns(min(len(features), 4))
        
        for i, (feature_name, feature_data) in enumerate(features.items()):
            with cols[i % 4]:
                st.metric(
                    label=feature_name,
                    value=f"{feature_data.mean():.3f}",
                    delta=f"范围: {feature_data.min():.3f} - {feature_data.max():.3f}"
                )

    def prepare_data_for_model(self, data, data_type):
        """准备数据用于模型推理"""
        try:
            # 检查是否有原始H5样本数据
            if 'original_samples' in data:
                # H5数据已经是正确的3D格式 (n_samples, 30, n_features)
                sequences = data['original_samples']
                sample_info = data['sample_info']
                n_samples = sample_info['n_samples']
                
                # 创建序列时间信息
                sequence_times = []
                for i in range(n_samples):
                    start_time = i * 30
                    end_time = (i + 1) * 30
                    sequence_times.append((start_time, end_time))
                
                # 数据预处理
                if data_type == 'battery':
                    # 电池数据使用StandardScaler
                    scaler = StandardScaler()
                    original_shape = sequences.shape
                    sequences_flat = sequences.reshape(-1, sequences.shape[-1])
                    sequences_scaled = scaler.fit_transform(sequences_flat)
                    sequences = sequences_scaled.reshape(original_shape)
                else:
                    # 飞行数据使用MinMaxScaler
                    scaler = MinMaxScaler()
                    original_shape = sequences.shape
                    sequences_flat = sequences.reshape(-1, sequences.shape[-1])
                    sequences_scaled = scaler.fit_transform(sequences_flat)
                    sequences = sequences_scaled.reshape(original_shape)
                return sequences, sequence_times, scaler
            
            else:
                # 旧的CSV数据处理逻辑（保留兼容性）
                time_data = data['time']
                features = data['features']
                
                # 根据数据类型获取期望的特征
                if data_type == 'battery':
                    expected_features = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h', 
                                       'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
                    expected_feature_count = 7
                elif data_type == 'flight':
                    expected_features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                                       'velocity', 'acceleration', 'altitude']
                    expected_feature_count = 9
                else:
                    return None
                
                # 检查特征完整性
                available_features = [f for f in expected_features if f in features]
                if len(available_features) < expected_feature_count:
                    st.warning(f"数据特征不完整，期望{expected_feature_count}个特征，实际{len(available_features)}个")
                    return None
                
                # 构建特征矩阵
                feature_matrix = np.column_stack([features[f] for f in available_features])
                
                # 按30秒窗口分割数据
                sequence_length = 30
                sequences = []
                sequence_times = []
                
                time_min, time_max = time_data.min(), time_data.max()
                current_time = time_min
                
                while current_time + sequence_length <= time_max:
                    # 找到这个30秒窗口的数据
                    window_mask = (time_data >= current_time) & (time_data < current_time + sequence_length)
                    window_data = feature_matrix[window_mask]
                    
                    if len(window_data) >= sequence_length:
                        # 如果数据点超过30个，进行重采样
                        if len(window_data) > sequence_length:
                            indices = np.linspace(0, len(window_data)-1, sequence_length, dtype=int)
                            window_data = window_data[indices]
                        
                        sequences.append(window_data)
                        sequence_times.append((current_time, current_time + sequence_length))
                    
                    current_time += sequence_length
                
                if not sequences:
                    st.warning("无法创建30秒序列，数据可能不足")
                    return None
                
                # 转换为numpy array
                sequences = np.array(sequences)
                
                # 数据预处理
                if data_type == 'battery':
                    # 电池数据使用StandardScaler
                    scaler = StandardScaler()
                    original_shape = sequences.shape
                    sequences_flat = sequences.reshape(-1, sequences.shape[-1])
                    sequences_scaled = scaler.fit_transform(sequences_flat)
                    sequences = sequences_scaled.reshape(original_shape)
                else:
                    # 飞行数据使用MinMaxScaler
                    scaler = MinMaxScaler()
                    original_shape = sequences.shape
                    sequences_flat = sequences.reshape(-1, sequences.shape[-1])
                    sequences_scaled = scaler.fit_transform(sequences_flat)
                    sequences = sequences_scaled.reshape(original_shape)
                
                st.info(f"使用CSV数据格式，从时间序列创建了{len(sequences)}个30秒样本")
                return sequences, sequence_times, scaler
            
        except Exception as e:
            st.error(f"数据预处理失败: {str(e)}")
            return None

    def perform_model_inference(self, sequences, model_info):
        """执行模型推理"""
        model_type = model_info.get('model_type', 'unknown')
        framework = model_info.get('framework', 'Unknown')
        model_object = st.session_state.model.get('object')
        
        if model_object is None:
            st.error("模型对象为空，请检查模型文件")
            return None, None
        
        if framework == "PyTorch":
            try:
                # 检查加载的对象类型
                if isinstance(model_object, dict):
                    # 这是一个state dict，需要先初始化模型架构
                    # st.info(f"{model_object['model_state_dict'].keys()}")
                    model_object = model_object['model_state_dict']
                    # 根据模型类型初始化对应的模型架构
                    if model_type == 'battery':
                        from src.models.battery_cnn_lstm import BatteryAnomalyNet
                        # 使用BatteryAnomalyNet的默认配置
                        config_dict = {
                            'sequence_length': 30,
                            'input_features': 7,
                            'num_classes': 2,
                            'cnn_channels': [16, 32, 64],
                            'lstm_hidden': 64,
                            'attention_heads': 2,
                            'classifier_hidden': [32],
                            'dropout_rate': 0.5
                        }
                        model = BatteryAnomalyNet(config_dict)
                    elif model_type == 'flight':
                        from src.models.flight_cnn_lstm import FlightAnomalyNet
                        model_dict = {                            'num_classes': 2,
                            'cnn_channels': [96, 192, 384],
                            'lstm_hidden': 256,
                            'attention_heads': 8,
                            'classifier_hidden': [128, 64]
                        }
                        model = FlightAnomalyNet(model_dict)
                    else:
                        st.error(f"未知的模型类型: {model_type}")
                        return None, None
                    
                    # 加载state dict
                    try:
                        model.load_state_dict(model_object)
                        st.success(f"成功加载{model_type}模型权重")
                    except Exception as e:
                        st.error(f"加载模型权重失败: {str(e)}")
                        return None, None
                    
                    # 设置为评估模式
                    model.eval()
                    
                    # 执行推理
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(sequences)
                        outputs = model(input_tensor)
                        
                        if isinstance(outputs, torch.Tensor):
                            if outputs.dim() == 2 and outputs.shape[1] == 2:
                                # 二分类输出
                                probabilities = torch.softmax(outputs, dim=1)
                                confidences = probabilities[:, 1].numpy()  # 异常类别的概率
                            else:
                                confidences = torch.sigmoid(outputs).squeeze().numpy()
                        else:
                            confidences = outputs
                        
                        anomaly_threshold = 0.5
                        anomaly_predictions = confidences > anomaly_threshold

                        # 显示模型性能统计
                        if 'original_labels' in st.session_state.data:
                            st.info("test")
                            true_labels = st.session_state.data['original_labels'][:len(anomaly_predictions)]
                            true_anomaly_ratio = np.mean(true_labels)
                            pred_anomaly_ratio = np.mean(anomaly_predictions)
                            accuracy = np.mean(anomaly_predictions == true_labels)
                            st.info(f"真实异常比例: {true_anomaly_ratio:.1%}，正常比例: {1-true_anomaly_ratio:.1%}")
                            st.info(f"模型预测异常比例: {pred_anomaly_ratio:.1%}，正常比例: {1-pred_anomaly_ratio:.1%}")
                            st.info(f"模型准确率: {accuracy:.1%}")
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(true_labels, anomaly_predictions)
                            st.write("混淆矩阵（真实/预测）: 0=正常, 1=异常")
                            st.write(cm)
                        
                        return anomaly_predictions, confidences
                        
                elif hasattr(model_object, 'eval'):
                    # 如果是完整的模型对象
                    st.info("检测到完整的PyTorch模型对象")
                    model_object.eval()
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(sequences)
                        outputs = model_object(input_tensor)
                        if isinstance(outputs, torch.Tensor):
                            if outputs.dim() == 2 and outputs.shape[1] == 2:
                                # 二分类输出
                                probabilities = torch.softmax(outputs, dim=1)
                                confidences = probabilities[:, 1].numpy()  # 异常类别的概率
                            else:
                                confidences = torch.sigmoid(outputs).squeeze().numpy()
                        else:
                            confidences = outputs
                        
                        anomaly_threshold = 0.5
                        anomaly_predictions = confidences > anomaly_threshold

                        if 'original_labels' in st.session_state.data:
                            true_labels = st.session_state.data['original_labels'][:len(anomaly_predictions)]
                            true_anomaly_ratio = np.mean(true_labels)
                            pred_anomaly_ratio = np.mean(anomaly_predictions)
                            accuracy = np.mean(anomaly_predictions == true_labels)
                            st.info(f"真实异常比例: {true_anomaly_ratio:.1%}，正常比例: {1-true_anomaly_ratio:.1%}")
                            st.info(f"模型预测异常比例: {pred_anomaly_ratio:.1%}，正常比例: {1-pred_anomaly_ratio:.1%}")
                            st.info(f"模型准确率: {accuracy:.1%}")
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(true_labels, anomaly_predictions)
                            st.write("混淆矩阵（真实/预测）: 0=正常, 1=异常")
                            st.write(cm)
                        
                        return anomaly_predictions, confidences
                else:
                    st.error(f"无法识别的PyTorch模型格式: {type(model_object)}")
                    return None, None
                
            except Exception as e:
                st.error(f"PyTorch模型推理失败: {str(e)}")
                return None, None
        else:
            st.error(f"框架 {framework} 暂不支持")
            return None, None

    def generate_anomaly_detection(self, time_data):
        if not st.session_state.model_loaded or not st.session_state.data_loaded:
            st.error("模型或数据未加载")
            return []
        
        data = st.session_state.data
        data_type = st.session_state.data_type
        model_info = st.session_state.model_info
        
        # 准备数据
        prepared_data = self.prepare_data_for_model(data, data_type)
        if prepared_data is None:
            return []
        
        sequences, sequence_times, scaler = prepared_data
        
        anomaly_predictions, confidences = self.perform_model_inference(sequences, model_info)
        if anomaly_predictions is None:
            return []
        
        anomaly_regions = []
        for i, (is_anomaly, confidence) in enumerate(zip(anomaly_predictions, confidences)):
            if is_anomaly:
                start_time, end_time = sequence_times[i]
                anomaly_regions.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': float(confidence),
                    'sequence_id': i
                })
        
        return anomaly_regions

    def detect_model_type(self, filename):
        """根据文件名前缀检测模型类型"""
        filename_lower = filename.lower()
        if filename_lower.startswith('battery_'):
            return 'battery'
        elif filename_lower.startswith('flight_'):
            return 'flight'
        else:
            return 'unknown'
    
    def validate_model_data_compatibility(self, model_type, data_type):
        """验证模型和数据的兼容性"""
        if model_type == 'unknown' or data_type == 'unknown':
            return False, "模型类型或数据类型未知"
        
        if model_type != data_type:
            return False, f"模型类型({model_type})与数据类型({data_type})不匹配"
        
        return True, "兼容"

    def load_model(self, uploaded_model):
        # 添加空值检查
        if uploaded_model is None:
            st.error("请先选择模型文件")
            return
        
        with st.spinner('加载模型...'):
            try:
                file_extension = Path(uploaded_model.name).suffix.lower()
                
                # 检测模型类型
                model_type = self.detect_model_type(uploaded_model.name)
                
                model_info = {
                    "filename": uploaded_model.name,
                    "filesize": f"{uploaded_model.size / 1024:.2f} KB",
                    "filetype": file_extension,
                    "model_type": model_type
                }
                
                # 根据文件类型加载不同的模型
                model_object = None
                if file_extension in ['.pkl', '.joblib']:
                    model_info["framework"] = "Scikit-learn"
                    try:
                        if file_extension == '.pkl':
                            model_object = pickle.loads(uploaded_model.getvalue())
                        else:
                            model_object = joblib.loads(uploaded_model.getvalue())
                    except Exception as e:
                        st.warning(f"无法加载模型对象: {str(e)}，将使用模拟模式")
                        model_object = None
                elif file_extension == '.h5':
                    model_info["framework"] = "TensorFlow/Keras"
                    # H5模型需要特殊处理，这里先保存数据
                    model_object = uploaded_model.getvalue()
                elif file_extension in ['.pth', '.pt']:
                    model_info["framework"] = "PyTorch"
                    try:
                        # 尝试加载PyTorch模型
                        import io
                        model_object = torch.load(io.BytesIO(uploaded_model.getvalue()), map_location='cpu')
                    except Exception as e:
                        st.warning(f"无法加载PyTorch模型: {str(e)}，将使用模拟模式")
                        model_object = uploaded_model.getvalue()
                elif file_extension == '.onnx':
                    model_info["framework"] = "ONNX"
                    model_object = uploaded_model.getvalue()
                else:
                    model_info["framework"] = "Unknown"
                    model_object = uploaded_model.getvalue()
                
                # 验证与数据的兼容性
                if st.session_state.data_loaded:
                    is_compatible, compatibility_msg = self.validate_model_data_compatibility(
                        model_type, st.session_state.data_type
                    )
                    model_info["compatibility"] = compatibility_msg
                    
                    if not is_compatible:
                        st.error(f"模型与数据不兼容: {compatibility_msg}")
                        st.error("请确保模型文件名以正确的前缀开头（battery_ 或 flight_）")
                        return
                    else:
                        st.success(f"模型与数据兼容性验证通过: {compatibility_msg}")
                
                st.session_state.model = {
                    'name': uploaded_model.name,
                    'type': file_extension,
                    'model_type': model_type,
                    'object': model_object,
                    'data': uploaded_model.getvalue()
                }
                st.session_state.model_info = model_info
                st.session_state.model_loaded = True
                
                st.success(f"模型 {uploaded_model.name} 加载成功！")
                
            except Exception as e:
                st.error(f"模型加载失败: {str(e)}")
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.session_state.model_info = {}

    def run(self):
        self.create_header()
        col1, col2 = st.columns([1, 2])
        with col1:
            self.create_control_panel()
        with col2:
            self.create_chart_area()

def main():
    app = BatteryAnalysisApp()
    app.run()

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    main() 