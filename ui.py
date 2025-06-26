import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
from pathlib import Path

st.set_page_config(
    page_title="电池健康管理-DEMO", 
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
            # 🟢🔴
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
                type=['csv', 'xlsx'],
                help="支持CSV和Excel格式（battery_开头为电池数据，flight_开头为飞行数据）"
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
            
            # 模型文件选择
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
                with st.spinner('正在执行模型识别...'):
                    time.sleep(2)  # 模拟模型推理时间
                    if hasattr(st.session_state, 'data'):
                        st.session_state.data['anomaly_regions'] = self.generate_anomaly_detection(st.session_state.data['time'])
                        st.session_state.model_detection_completed = True
                        st.success("模型识别完成！")
                        st.rerun()
            
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
            
            # 添加异常标签显示控制
            self.create_anomaly_label_control()
            
            # 添加时间范围选择功能
            self.create_time_range_selector(time_min, time_max)
            
            # 根据选择的时间范围过滤数据
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
        """创建异常标签显示控制"""
        st.markdown("### 异常标签显示")
        
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
                
                if file_extension == '.csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error(f"不支持的文件格式: {file_extension}")
                    return
                
                if df.empty:
                    st.error("文件为空或无法读取")
                    return
                
                st.success(f"成功读取数据，形状: {df.shape}")
                
                with st.expander("数据预览"):
                    st.dataframe(df.head())
                
                # 检测数据类型
                data_type = self.detect_data_type(uploaded_file.name)
                st.session_state.data_type = data_type
                
                processed_data = self.process_data_by_type(df, data_type)
                if processed_data is None:
                    return
                
                st.session_state.data = processed_data
                st.session_state.data_loaded = True
                st.session_state.model_detection_completed = False
                st.session_state.selected_time_range = None  # 重置时间选择
                    
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
                st.session_state.data_loaded = False
                st.session_state.data = None

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
            'EnergyCharge_W_h': 'green',
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
        
        # 添加模型检测的异常区域（红色）
        if st.session_state.model_detection_completed and 'anomaly_regions' in data:
            for region in data['anomaly_regions']:
                start, end = region['start'], region['end']
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
                    
                    fig.add_trace(go.Scatter(
                        x=[start, start, end, end, start],
                        y=[y_min, y_max, y_max, y_min, y_min],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        line=dict(color="red", width=1),
                        showlegend=False,
                        hoverinfo="skip"
                    ), row=row, col=col)
        
        # 添加数据中的异常标签（黄色）
        if st.session_state.show_data_labels and 'data_labels' in data:
            for label in data['data_labels']:
                start, end = label['start'], label['end']
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
                    
                    fig.add_trace(go.Scatter(
                        x=[start, start, end, end, start],
                        y=[y_min, y_max, y_max, y_min, y_min],
                        fill="toself",
                        fillcolor="rgba(255, 255, 0, 0.3)",
                        line=dict(color="yellow", width=1),
                        showlegend=False,
                        hoverinfo="skip"
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
        
        # 3. 3D轨迹
        if 'x' in features and 'y' in features and 'z' in features:
            # 3D轨迹 - 根据是否显示标签分段着色
            if False and st.session_state.show_data_labels and 'data_labels' in data and len(data['data_labels']) > 0:
                # 为轨迹分段着色，标记异常区域
                current_idx = 0
                for i, label in enumerate(data['data_labels']):
                    # 添加正常段
                    label_start_idx = np.searchsorted(time_data, label['start'])
                    if current_idx < label_start_idx:
                        fig.add_trace(go.Scatter3d(
                            x=features['x'][current_idx:label_start_idx],
                            y=features['y'][current_idx:label_start_idx],
                            z=features['z'][current_idx:label_start_idx],
                            mode='lines',
                            name='正常轨迹' if i == 0 else None,
                            line=dict(color='blue', width=1),
                            showlegend=i == 0
                        ), row=2, col=1)
                    
                    # 添加异常段
                    label_end_idx = np.searchsorted(time_data, label['end'])
                    if label_start_idx < label_end_idx:
                        fig.add_trace(go.Scatter3d(
                            x=features['x'][label_start_idx:label_end_idx],
                            y=features['y'][label_start_idx:label_end_idx],
                            z=features['z'][label_start_idx:label_end_idx],
                            mode='lines',
                            name='异常轨迹' if i == 0 else None,
                            line=dict(color='yellow', width=1),
                            showlegend=i == 0
                        ), row=2, col=1)
                    current_idx = label_end_idx
                
                # 添加最后一段正常轨迹
                if current_idx < len(time_data):
                    fig.add_trace(go.Scatter3d(
                        x=features['x'][current_idx:],
                        y=features['y'][current_idx:],
                        z=features['z'][current_idx:],
                        mode='lines',
                        name=None,
                        line=dict(color='blue', width=1),
                        showlegend=False
                    ), row=2, col=1)
            else:
                fig.add_trace(go.Scatter3d(
                    x=features['x'], y=features['y'], z=features['z'],
                    mode='lines',
                    name='3D轨迹',
                    line=dict(color='blue', width=1),
                ), row=2, col=1)
            subplot_has_data[(2, 1)] = True
        
        # 4. X, Y, Z 位置时间序列
        pos_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
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
            for region in data['anomaly_regions']:
                start, end = region['start'], region['end']
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
                        
                        fig.add_trace(go.Scatter(
                            x=[start, start, end, end, start],
                            y=[y_min, y_max, y_max, y_min, y_min],
                            fill="toself",
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            line=dict(color="red", width=1),
                            showlegend=False,
                            hoverinfo="skip"
                        ), row=row, col=col)
        
        # 添加数据中的异常标签（黄色）
        if st.session_state.show_data_labels and 'data_labels' in data:
            for label in data['data_labels']:
                start, end = label['start'], label['end']
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
                        
                        fig.add_trace(go.Scatter(
                            x=[start, start, end, end, start],
                            y=[y_min, y_max, y_max, y_min, y_min],
                            fill="toself",
                            fillcolor="rgba(255, 255, 0, 0.3)",
                            line=dict(color="yellow", width=1),
                            showlegend=False,
                            hoverinfo="skip"
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

    def generate_anomaly_detection(self, time_data):
        anomaly_regions = []
        
        segment_length = 3000  # 序列长度500s
        time_min, time_max = time_data.min(), time_data.max()
        
        current_time = time_min
        segment_id = 1
        
        while current_time < time_max:
            segment_end = min(current_time + segment_length, time_max)
            
            # 随机决定这个段是否异常 (20%概率为异常)
            if np.random.random() < 0.1:
                # 在这个段内随机选择异常的子区间
                anomaly_duration = np.random.uniform(30, 60)  # 异常持续时间10-50秒
                anomaly_start = current_time + np.random.uniform(0, max(0, segment_length - anomaly_duration))
                anomaly_end = min(anomaly_start + anomaly_duration, segment_end)
                
                anomaly_regions.append({
                    'start': anomaly_start,
                    'end': anomaly_end,
                    'segment_id': segment_id,
                    'confidence': np.random.uniform(0.7, 0.95)  # 检测置信度
                })
                
            
            current_time += segment_length
            segment_id += 1
        
        return anomaly_regions

    def load_model(self, uploaded_model):
        with st.spinner('加载模型...'):
            time.sleep(1) 
            try:
                file_extension = Path(uploaded_model.name).suffix.lower()
                
                model_info = {
                    "filename": uploaded_model.name,
                    "filesize": f"{uploaded_model.size / 1024:.2f} KB",
                    "filetype": file_extension,
                }
                
                # 根据文件类型模拟不同的模型加载
                if file_extension in ['.pkl', '.joblib']:
                    model_info["framework"] = "Scikit-learn"
                elif file_extension == '.h5':
                    model_info["framework"] = "TensorFlow/Keras"
                elif file_extension in ['.pth', '.pt']:
                    model_info["framework"] = "PyTorch"
                elif file_extension == '.onnx':
                    model_info["framework"] = "ONNX"
                else:
                    model_info["framework"] = "Unknown"
                
                # 模拟模型加载过程
                st.session_state.model = {
                    'name': uploaded_model.name,
                    'type': file_extension,
                    'data': uploaded_model.getvalue()  # TODO 在实际应用中，这里会是真正的模型对象
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
    main() 