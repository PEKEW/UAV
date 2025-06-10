import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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
        if 'selected_range' not in st.session_state:
            st.session_state.selected_range = None
        if 'abnormal_marking_mode' not in st.session_state:
            st.session_state.abnormal_marking_mode = False
        if 'abnormal_regions' not in st.session_state:
            st.session_state.abnormal_regions = []
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        if 'selected_feature_name' not in st.session_state:
            st.session_state.selected_feature_name = None
        if 'model_detection_completed' not in st.session_state:
            st.session_state.model_detection_completed = False

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
                help="支持CSV和Excel格式"
            )
            
            if st.button("导入数据", use_container_width=True, disabled=not (uploaded_file is not None and st.session_state.current_status == 'offline')):
                    self.load_data(uploaded_file)
            
            if st.session_state.data_loaded:
                st.success("数据已导入")
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
            
            st.markdown("---")
            
            st.markdown("### 选择分析特征")
            
            if st.session_state.data_loaded and hasattr(st.session_state, 'available_features'):
                available_features = st.session_state.available_features
                current_feature = st.session_state.get('current_feature', available_features[0] if available_features else None)
                
                if available_features:
                    # 创建特征选择下拉框
                    feature_choice = st.selectbox(
                        "选择要分析的主要特征",
                        available_features,
                        format_func=lambda x: x[0],
                        index=available_features.index(current_feature) if current_feature in available_features else 0,
                        key="feature_selector"
                    )
                    
                    if feature_choice != current_feature:
                        if st.button("切换特征", key="switch_feature"):
                            self.switch_feature(feature_choice)
                            st.rerun()
                    
                    st.info(f"当前分析特征: {feature_choice[0]} ({feature_choice[1]})")
                    
                    st.markdown("---")
                    if st.button("重新检测异常", key="redetect_anomalies"):
                        if hasattr(st.session_state, 'data'):
                            st.session_state.data['anomaly_regions'] = self.generate_anomaly_detection(st.session_state.data['time'])
                            st.success("异常检测已更新")
                            st.rerun()
                else:
                    st.warning("请先上传数据文件")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_chart_area(self):
        st.markdown("##  数据可视化与异常检测")
        if st.session_state.data_loaded and st.session_state.data is not None:
            data = st.session_state.data
            
            total_points = len(data['time'])
            time_min, time_max = data['time'].min(), data['time'].max()
            total_duration = time_max - time_min
            
            st.info(f"数据总量: {total_points} 个数据点 | 时间范围: {time_min:.1f}s - {time_max:.1f}s | 总时长: {total_duration:.1f}s")
            
            MAX_TIME_RANGE = 100000
            
            if st.session_state.selected_range is not None:
                start_time, end_time = st.session_state.selected_range
                
                selected_duration = end_time - start_time
                if selected_duration > MAX_TIME_RANGE:
                    end_time = start_time + MAX_TIME_RANGE
                    st.warning(f"选择的时间范围超过{MAX_TIME_RANGE}秒限制，已自动截取到 {start_time:.1f}s - {end_time:.1f}s")
                
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                time_data = data['time'][mask]
                true_values = data['true_values'][mask]
                actual_duration = end_time - start_time
                range_points = len(time_data)
                
                display_info = f"显示范围: {start_time:.1f}s - {end_time:.1f}s (时长: {actual_duration:.1f}s, {range_points} 个点)"
            else:
                if total_duration > MAX_TIME_RANGE:
                    max_time = time_min + MAX_TIME_RANGE
                    mask = data['time'] <= max_time
                    time_data = data['time'][mask]
                    true_values = data['true_values'][mask]
                    actual_points = len(time_data)
                    display_info = f"显示: 前{MAX_TIME_RANGE}秒 ({time_min:.1f}s - {max_time:.1f}s, {actual_points} 个点) | 总时长: {total_duration:.1f}s"
                else:
                    time_data = data['time']
                    true_values = data['true_values']
                    display_info = f"显示全部: {total_duration:.1f}s ({total_points} 个点)"
            
            st.info(f"{display_info}")
            
            if len(time_data) == 0:
                st.warning("选择的时间范围内没有数据")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=true_values,
                mode='lines',
                name='传感器数据',
                line=dict(color='#1f77b4', width=2),
                opacity=0.8,
                hovertemplate='时间: %{x:.1f}s<br>数值: %{y:.4f}<extra></extra>'
            ))
            
            if st.session_state.model_detection_completed and 'anomaly_regions' in data:
                for i, region in enumerate(data['anomaly_regions']):
                    start, end = region['start'], region['end']
                    # 只显示在当前时间范围内的异常区域
                    if start <= time_data.max() and end >= time_data.min():
                        fig.add_vrect(
                            x0=max(start, time_data.min()),
                            x1=min(end, time_data.max()),
                            fillcolor=f"rgba(255, 0, 0, 0.3)",
                            layer="below",
                            line_width=1,
                            line_color="red",
                            annotation_text=f"异常区域 {i+1}<br>置信度: {region['confidence']:.2f}",
                            annotation_position="top left"
                        )
            
            for i, (start, end) in enumerate(st.session_state.abnormal_regions):
                if start >= time_data.min() and end <= time_data.max():
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="rgba(255, 165, 0, 0.4)",
                        layer="below",
                        line_width=2,
                        line_color="orange",
                        annotation_text=f"手动标记 {i+1}",
                        annotation_position="top left"
                    )
            
            if st.session_state.abnormal_marking_mode and hasattr(st.session_state, 'preview_start') and hasattr(st.session_state, 'preview_end'):
                fig.add_vrect(
                    x0=st.session_state.preview_start,
                    x1=st.session_state.preview_end,
                    fillcolor="rgba(255, 255, 0, 0.3)",
                    layer="below",
                    line_width=2,
                    line_color="orange",
                    annotation_text="预览区域",
                    annotation_position="top left"
                )
            
            feature_name = self.get_feature_display_name(data)
            
            fig.update_layout(
                title=f'{feature_name} 异常检测',
                xaxis_title='时间 (秒)',
                yaxis_title=feature_name,
                hovermode='x unified',
                height=500,
                showlegend=True,
                xaxis=dict(
                    range=[time_data.min(), time_data.max()],
                    showgrid=True
                ),
                yaxis=dict(
                    showgrid=True
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            
            self.show_metrics(true_values, len(time_data))
            
            st.markdown("### 数据选择与分析")
            range_col1, range_col2, range_col3, range_col4 = st.columns(4)
            
            with range_col1:
                start_time = st.number_input(
                    "起始时间 (秒)", 
                    min_value=float(time_min), 
                    max_value=float(time_max), 
                    value=float(time_min),
                    step=1.0,
                    format="%.1f"
                )
            
            with range_col2:
                default_end = min(float(time_min) + 1000, float(time_max), start_time + MAX_TIME_RANGE)
                end_time = st.number_input(
                    "结束时间 (秒)", 
                    min_value=float(time_min), 
                    max_value=float(time_max), 
                    value=default_end,
                    step=1.0,
                    format="%.1f"
                )
            
            with range_col3:
                if st.button("应用选择"):
                    if start_time < end_time:
                        if end_time - start_time > MAX_TIME_RANGE:
                            st.warning(f"时间范围超过{MAX_TIME_RANGE}秒限制，将自动截取到{MAX_TIME_RANGE}秒")
                            end_time = start_time + MAX_TIME_RANGE
                        
                        st.session_state.selected_range = (start_time, end_time)
                        st.success(f"已选择范围: {start_time:.1f}s - {end_time:.1f}s (时长: {end_time-start_time:.1f}s)")
                        st.rerun()
                    else:
                        st.error("时间范围非法：起始时间必须小于结束时间！")
            
            with range_col4:
                if st.button("重置范围"):
                    st.session_state.selected_range = None
                    st.success("已重置为显示全部数据")
                    st.rerun()
            
            if total_duration > MAX_TIME_RANGE:
                st.warning(f"数据时间跨度较大 ({total_duration:.1f}秒)，仅显示前{MAX_TIME_RANGE}秒。")
            
            st.markdown("---")
            st.markdown("### 异常标记控制")
            
            abnormal_col1, abnormal_col2, abnormal_col3 = st.columns(3)
            
            with abnormal_col1:
                if st.button("开始标记", use_container_width=True):
                    st.session_state.abnormal_marking_mode = True
                    st.rerun()
            
            with abnormal_col2:
                if st.session_state.abnormal_marking_mode:
                    if st.button("更新异常区间", use_container_width=True):
                        if hasattr(st.session_state, 'preview_start') and hasattr(st.session_state, 'preview_end'):
                            st.session_state.preview_start = st.session_state.preview_start
                            st.session_state.preview_end = st.session_state.preview_end
                            st.success(f"已更新预览区域: {st.session_state.preview_start:.1f}s - {st.session_state.preview_end:.1f}s")
                            st.rerun()
                        else:
                            st.warning("请先在滑动条上选择时间范围")
                else:
                    if st.button("停止标记", use_container_width=True):
                        st.session_state.abnormal_marking_mode = False
                        if hasattr(st.session_state, 'preview_start'):
                            delattr(st.session_state, 'preview_start')
                        if hasattr(st.session_state, 'preview_end'):
                            delattr(st.session_state, 'preview_end')
                        st.rerun()
            
            with abnormal_col3:
                if st.button("清除标记", use_container_width=True):
                    st.session_state.abnormal_regions = []
                    st.success("已清除所有异常标记")
                    st.rerun()
            
            if st.session_state.abnormal_marking_mode:
                st.markdown("#### 拖拽滑动条选择异常区域")
                time_min_display = float(time_data.min())
                time_max_display = float(time_data.max())
                
                abnormal_range = st.slider(
                    "选择异常时间",
                    min_value=time_min_display,
                    max_value=time_max_display,
                    value=(time_min_display, time_min_display + min(10.0, time_max_display - time_min_display)),
                    step=0.1,
                    key="abnormal_range_slider"
                )
                
                st.session_state.preview_start = abnormal_range[0]
                st.session_state.preview_end = abnormal_range[1]
                
                st.info(f"当前选择范围: {abnormal_range[0]:.1f}s - {abnormal_range[1]:.1f}s (时长: {abnormal_range[1] - abnormal_range[0]:.1f}s)")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("确认并结束标记"):
                        st.session_state.abnormal_regions.append(abnormal_range)
                        st.session_state.abnormal_marking_mode = False
                        if hasattr(st.session_state, 'preview_start'):
                            delattr(st.session_state, 'preview_start')
                        if hasattr(st.session_state, 'preview_end'):
                            delattr(st.session_state, 'preview_end')
                        st.success(f"已标记异常区域: {abnormal_range[0]:.1f}s - {abnormal_range[1]:.1f}s")
                        st.rerun()
                
                with col_b:
                    if st.button("添加此区间并继续"):
                        st.session_state.abnormal_regions.append(abnormal_range)
                        st.success(f"已添加异常区域: {abnormal_range[0]:.1f}s - {abnormal_range[1]:.1f}s")
                        st.rerun()
            
            # 显示已标记的异常区域
            if st.session_state.abnormal_regions:
                st.markdown("#### 已标记的异常区域")
                for i, (start, end) in enumerate(st.session_state.abnormal_regions):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"异常区域 {i+1}: {start:.1f}s - {end:.1f}s (持续时间: {end-start:.1f}s)")
                    with col2:
                        if st.button("删除", key=f"delete_abnormal_{i}"):
                            st.session_state.abnormal_regions.pop(i)
                            st.rerun()
        
        else:
            st.info("请先上传数据文件开始分析")
    
    def get_feature_display_name(self, data):
        """获取特征的显示名称"""
        if hasattr(st.session_state, 'selected_feature_name'):
            return st.session_state.selected_feature_name
        
        values = data['true_values']
        value_range = values.max() - values.min()
        avg_value = values.mean()
        
        if 2.5 <= avg_value <= 4.5 and value_range < 2:
            return "电池电压 (V)"
        elif abs(avg_value) > 100 and value_range > 10:
            return "电流 (mA)"
        elif 10 <= avg_value <= 50:
            return "温度 (°C)"
        else:
            return "特征值"
    
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
                
                column_mapping = self.detect_columns(df)
                
                if not column_mapping:
                    st.error("请选择要分析的数据特征")
                    return
                
                processed_data = self.process_data(df, column_mapping)
                if processed_data is None:
                    return
                
                st.session_state.data = processed_data
                st.session_state.data_loaded = True
                
                st.session_state.selected_range = None
                st.session_state.abnormal_regions = []
                st.session_state.abnormal_marking_mode = False
                if hasattr(st.session_state, 'preview_start'):
                    delattr(st.session_state, 'preview_start')
                if hasattr(st.session_state, 'preview_end'):
                    delattr(st.session_state, 'preview_end')
                
                st.session_state.model_detection_completed = False
                
                st.info("已清除之前的选择范围和手动标记区域")
                    
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
                st.session_state.data_loaded = False
                st.session_state.data = None

    def detect_columns(self, df):
        column_mapping = {}
        columns = df.columns.tolist()
        
        
        time_keywords = ['time_s', 'time', 'timestamp', '时间', 'Time', 'TIME', 't']
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in time_keywords):
                column_mapping['time'] = col
                break
        
        voltage_keywords = ['ecell_v', 'voltage', 'volt', 'v', '电压', 'Ecell_V', 'cell_voltage']
        current_keywords = ['i_ma', 'current', 'amp', 'i', '电流', 'I_mA', 'I_A']
        temperature_keywords = ['temperature', 'temp', '温度', 'T']
        
        voltage_col = None
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in voltage_keywords):
                voltage_col = col
                break
        
        current_col = None
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in current_keywords):
                current_col = col
                break
        
        temperature_col = None
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in temperature_keywords):
                temperature_col = col
                break
        
        if 'time' not in column_mapping:
            column_mapping['time'] = 'index'
        
        available_features = []
        if voltage_col:
            available_features.append(('电压 (Voltage)', voltage_col, '电池电压 (V)'))
        if current_col:
            available_features.append(('电流 (Current)', current_col, '电流 (mA)'))
        if temperature_col:
            available_features.append(('温度 (Temperature)', temperature_col, '温度 (°C)'))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in [voltage_col, current_col, temperature_col, column_mapping.get('time')]:
                available_features.append((f'{col}', col, f'{col}'))
        
        st.session_state.available_features = available_features
        st.session_state.original_df = df.copy() 
        
        if voltage_col:
            column_mapping['true_values'] = voltage_col
            st.session_state.selected_feature_name = '电池电压 (V)'
            st.session_state.current_feature = ('电压 (Voltage)', voltage_col, '电池电压 (V)')
        elif available_features:
            column_mapping['true_values'] = available_features[0][1]
            st.session_state.selected_feature_name = available_features[0][2]
            st.session_state.current_feature = available_features[0]
        else:
            st.error("未找到可用的数值特征列")
            return None
        
        st.success(f"默认使用特征: {st.session_state.selected_feature_name}")
        return column_mapping

    def process_data(self, df, column_mapping):
        try:
            processed_data = {}
            
            if column_mapping['time'] == 'index':
                processed_data['time'] = np.arange(len(df))
            else:
                time_col = df[column_mapping['time']]
                processed_data['time'] = pd.to_numeric(time_col, errors='coerce').values
            
            true_values = pd.to_numeric(df[column_mapping['true_values']], errors='coerce')
            if true_values.isna().all():
                st.error(f"特征列 '{column_mapping['true_values']}' 无法转换为数值")
                return None
            processed_data['true_values'] = true_values.values
            
            valid_mask = ~(np.isnan(processed_data['time']) | np.isnan(processed_data['true_values']))
            for key in processed_data:
                processed_data[key] = processed_data[key][valid_mask]
            
            if len(processed_data['time']) == 0:
                st.error("处理后的数据为空")
                return None
            
            sort_indices = np.argsort(processed_data['time'])
            for key in processed_data:
                processed_data[key] = processed_data[key][sort_indices]
            
            st.success(f"成功处理 {len(processed_data['time'])} 个数据点")
            st.info(f"时间范围: {processed_data['time'].min():.1f}s - {processed_data['time'].max():.1f}s")
            st.info(f"特征值范围: {processed_data['true_values'].min():.4f} - {processed_data['true_values'].max():.4f}")
            
            return processed_data
            
        except Exception as e:
            st.error(f"数据处理失败: {str(e)}")
            return None

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

    
    def show_metrics(self, true_vals, data_count):
        st.markdown("### 数据统计信息")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.info(f"**数据点数量**: {data_count}")
        
        with stat_col2:
            st.info(f"**特征值范围**: [{true_vals.min():.3f}, {true_vals.max():.3f}]")
        
        if not st.session_state.model_detection_completed:
            st.info(f"加载模型并点击\"开始识别\"后将显示异常检测指标")
            return
        
        if not hasattr(st.session_state, 'data') or 'anomaly_regions' not in st.session_state.data:
            return
        
        anomaly_regions = st.session_state.data['anomaly_regions']
        
        total_anomaly_duration = sum((region['end'] - region['start']) for region in anomaly_regions)
        total_duration = st.session_state.data['time'].max() - st.session_state.data['time'].min()
        anomaly_rate = (total_anomaly_duration / total_duration) * 100 if total_duration > 0 else 0
        
        if st.session_state.selected_range is not None:
            start_time, end_time = st.session_state.selected_range
            current_anomalies = [r for r in anomaly_regions 
                               if r['start'] <= end_time and r['end'] >= start_time]
            current_anomaly_duration = sum((min(r['end'], end_time) - max(r['start'], start_time)) 
                                         for r in current_anomalies)
            current_duration = end_time - start_time
            current_anomaly_rate = (current_anomaly_duration / current_duration) * 100 if current_duration > 0 else 0
        else:
            current_anomalies = anomaly_regions
            current_anomaly_rate = anomaly_rate
        
        st.markdown("### 异常检测指标")
        
        metric_row1_col1, metric_row1_col2, metric_row1_col3 = st.columns(3)
        
        with metric_row1_col1:
            st.metric(
                label="异常数量",
                value=f"{len(current_anomalies)}",
                help="当前范围内检测到的异常数量"
            )
        
        with metric_row1_col2:
            st.metric(
                label="异常占比",
                value=f"{current_anomaly_rate:.2f}%",
                help="异常占总时间的百分比"
            )
        
        with metric_row1_col3:
            avg_confidence = np.mean([r['confidence'] for r in current_anomalies]) if current_anomalies else 0
            st.metric(
                label="置信度",
                value=f"{avg_confidence:.2f}",
                help="异常检测的置信度"
            )
        
        metric_row2_col1, metric_row2_col2, metric_row2_col3 = st.columns(3)
        
        with metric_row2_col1:
            st.metric(
                label="准确率",
                value="0%",
                help="准确率"
            )
        
        with metric_row2_col2:
            segments_analyzed = int((st.session_state.data['time'].max() - st.session_state.data['time'].min()) / 3000) + 1
            st.metric(
                label="段数",
                value=f"{segments_analyzed}",
                help="按3000s长度分析的序列段数"
            )
        
        with metric_row2_col3:
            model_info = st.session_state.model_info.get('framework', '未知框架')
            st.metric(
                label="模型",
                value=model_info,
                help="当前使用的模型框架"
            )

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

    def switch_feature(self, new_feature):
        try:
            if not hasattr(st.session_state, 'original_df'):
                st.error("原始数据不可用，请重新上传文件")
                return
            
            st.session_state.current_feature = new_feature
            st.session_state.selected_feature_name = new_feature[2]
            
            column_mapping = {
                'true_values': new_feature[1]
            }
            
            original_time_col = None
            for col in st.session_state.original_df.columns:
                time_keywords = ['time_s', 'time', 'timestamp', '时间', 'Time', 'TIME', 't']
                if any(keyword.lower() in col.lower() for keyword in time_keywords):
                    original_time_col = col
                    break
            column_mapping['time'] = original_time_col if original_time_col else 'index'
            
            processed_data = self.process_data(st.session_state.original_df, column_mapping)
            
            if processed_data is not None:
                st.session_state.data = processed_data
                
                st.session_state.selected_range = None
                st.session_state.abnormal_regions = []
                st.session_state.abnormal_marking_mode = False
                if hasattr(st.session_state, 'preview_start'):
                    delattr(st.session_state, 'preview_start')
                if hasattr(st.session_state, 'preview_end'):
                    delattr(st.session_state, 'preview_end')
                
                st.success(f"已切换到特征: {new_feature[0]}")
                st.info("已清除选择范围和手动标记区域")
            else:
                st.error("特征切换失败")
            
            st.session_state.model_detection_completed = False
            
        except Exception as e:
            st.error(f"特征切换失败: {str(e)}")

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