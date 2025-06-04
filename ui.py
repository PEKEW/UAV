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
    page_icon="🔋",
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

    def create_header(self):
        st.markdown('<h1 class="main-header">🔋 电池健康管理-DEMO</h1>', unsafe_allow_html=True)
    
    def create_control_panel(self):
        with st.container():
            st.markdown('<div class="control-panel">📋控制面板</div>', unsafe_allow_html=True)
            
            st.markdown("### 🔌 监测模式 ")
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
                st.markdown('<div class="status-online"> ⚠️ 没有实时监测！</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-offline">🟢 离线监测</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📁 数据选择")
            uploaded_file = st.file_uploader(
                "点击导入或拖拽数据文件到此处",
                type=['csv', 'xlsx'],
                help="支持CSV和Excel格式"
            )
            
            # if uploaded_file is not None and st.session_state.current_status == 'offline':
            if st.button("🔄 导入数据", use_container_width=True, disabled=not (uploaded_file is not None and st.session_state.current_status == 'offline')):
                    self.load_data(uploaded_file)
            
            if st.session_state.data_loaded:
                st.success("✅ 数据已导入")
            else:
                st.info("⏳ 未导入数据")
            
            st.markdown("---")
            
            st.markdown("### 🔍 异常识别")
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("模型选择", use_container_width=True, disabled=not st.session_state.data_loaded):
                    st.toast("暂未实现！", icon="🚧")
            
            with col4:
                if st.button("开始识别(务必确保模型类型和识别特征匹配)", use_container_width=True, disabled=not st.session_state.data_loaded):
                    st.toast("暂未实现！", icon="🚧")
            
            st.markdown("---")
            
            st.markdown("### ⚙️ 选择预测特征")
            
            col5, col6 = st.columns(2)
            with col5:
                if st.button("🔋 电压", use_container_width=True, disabled=not st.session_state.data_loaded):
                    st.toast("暂未实现！", icon="🚧")
            
            with col6:
                if st.button("⚡ 电流", use_container_width=True, disabled=not st.session_state.data_loaded):
                    st.toast("暂未实现！", icon="🚧")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_chart_area(self):
        # TODO 这里的范围是给一个缺省值，然后根据导入的数据调整范围
        st.markdown("## 📈 数据可视化")
        if st.session_state.data_loaded and st.session_state.data is not None:
            data = st.session_state.data
            if st.session_state.selected_range is not None:
                start_time, end_time = st.session_state.selected_range
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                time_data = data['time'][mask]
                true_values = data['true_values'][mask]
                predictions = data['predictions'][mask]
            else:
                time_data = data['time'][:500]
                true_values = data['true_values'][:500]
                predictions = data['predictions'][:500]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_data,
                y=true_values,
                mode='lines',
                name='True Value',
                line=dict(color='#1f77b4', width=2),
                opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                x=time_data,
                y=predictions,
                mode='lines',
                name='Prediction Value',
                line=dict(color='#ff7f0e', width=2),
                opacity=0.8
            ))
            # TODO   置信区间根据模型输出结果设置
            confidence_upper = predictions + 0.1
            confidence_lower = predictions - 0.1
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_data, time_data[::-1]]),
                y=np.concatenate([confidence_upper, confidence_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 204, 204, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            for i, (start, end) in enumerate(st.session_state.abnormal_regions):
                if start >= time_data.min() and end <= time_data.max():
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        layer="below",
                        line_width=0,
                        annotation_text=f"异常区域 {i+1}",
                        annotation_position="top left"
                    )
                # TODO 保存异常区域范围 用于后续的指标计算
            
            # 如果在标记模式下，添加预览区域
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
            
            if st.session_state.selected_range is not None:
                start_time, end_time = st.session_state.selected_range
                title = f'True Value vs Prediction Value Analysis (时间范围: {start_time}-{end_time})'
            else:
                title = 'True Value vs Prediction Value Analysis (默认显示前500个点)'
            
            fig.update_layout(
                title=title,
                xaxis_title='Time Steps',
                yaxis_title='Battery Voltage (V)',
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            st.markdown("### 🎯 数据选择与分析")
            range_col1, range_col2, range_col3, range_col4 = st.columns(4)
            with range_col1:
                start_time = st.number_input("起始时间", min_value=0, max_value=int(max(data['time'])), value=0)
            with range_col2:
                end_time = st.number_input("结束时间", min_value=0, max_value=int(max(data['time'])), value=100)
            with range_col3:
                if st.button("🎯 应用选择"):
                    if start_time < end_time:
                        st.session_state.selected_range = (start_time, end_time)
                        st.success(f"已选择范围: {start_time} - {end_time}")
                        st.rerun()
                    else:
                        st.error("时间范围非法：起始时间必须小于结束时间！")
            with range_col4:
                if st.button("🔄 重置范围"):
                    st.session_state.selected_range = None
                    st.success("已重置")
                    st.rerun()
            st.markdown("---")
            st.markdown("### ⚠️ 异常标记控制")
            
            abnormal_col1, abnormal_col2, abnormal_col3 = st.columns(3)
            
            with abnormal_col1:
                if st.button("🖱️ 开始标记", use_container_width=True):
                    st.session_state.abnormal_marking_mode = True
                    st.rerun()
            
            with abnormal_col2:
                if st.button("⏹️ 停止标记", use_container_width=True):
                    st.session_state.abnormal_marking_mode = False
                    if hasattr(st.session_state, 'preview_start'):
                        delattr(st.session_state, 'preview_start')
                    if hasattr(st.session_state, 'preview_end'):
                        delattr(st.session_state, 'preview_end')
                    st.rerun()
            
            with abnormal_col3:
                if st.button("🗑️ 清除标记", use_container_width=True):
                    st.session_state.abnormal_regions = []
                    st.success("已清除所有异常标记")
                    st.rerun()
            
            if st.session_state.abnormal_marking_mode:
                st.markdown("#### 🎯 拖拽滑动条选择异常区域")
                time_min = float(time_data.min())
                time_max = float(time_data.max())
                
                selected_range = st.slider(
                    "选择异常区域范围",
                    min_value=time_min,
                    max_value=time_max,
                    value=(time_min, time_min + (time_max - time_min) * 0.1),
                    step=1.0,
                    key="abnormal_range_slider"
                )
                
                st.session_state.preview_start = selected_range[0]
                st.session_state.preview_end = selected_range[1]
                
                st.info(f"当前选择范围: {selected_range[0]:.1f} - {selected_range[1]:.1f}")
                
                mark_col1, mark_col2 = st.columns(2)
                
                with mark_col1:
                    if st.button("➕ 添加异常标记", use_container_width=True):
                        new_region = (selected_range[0], selected_range[1])
                        if new_region not in st.session_state.abnormal_regions:
                            st.session_state.abnormal_regions.append(new_region)
                            st.success(f"已添加异常区域: {selected_range[0]:.1f} - {selected_range[1]:.1f}")
                            st.rerun()
                        else:
                            st.warning("该异常区域已存在！")
                
                with mark_col2:
                    if st.button("🔄 刷新预览", use_container_width=True):
                        st.rerun()
            
            if st.session_state.abnormal_regions:
                st.markdown("#### 📋 已标记异常区域")
                for i, (start, end) in enumerate(st.session_state.abnormal_regions):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"异常区域 {i+1}: {start:.1f} - {end:.1f}")
                    with col2:
                        if st.button("删除", key=f"delete_{i}"):
                            st.session_state.abnormal_regions.pop(i)
                            st.rerun()
            
            st.markdown("---")
            self.show_metrics(true_values, predictions, len(time_data))
        
        else:
            st.info("请先导入数据以查看可视化图表")
    
    def load_data(self, uploaded_file):
        with st.spinner('加载数据...'):
            time.sleep(1)  # 模拟加载延迟
            # self.get_data_from_file(uploaded_file)
            self.generate_sample_data()
            st.session_state.data_loaded = True
    
    def get_data_from_file(self, uploaded_file):
        raise NotImplementedError("Not implemented")
    
    def generate_sample_data(self):
        # INFO fake for sim
        time_steps = np.arange(0, 1000)
        targets = 3.7 + 0.3 * np.sin(time_steps * 0.01) + np.random.normal(0, 0.05, len(time_steps))
        predictions = targets + np.random.normal(0, 0.08, len(time_steps))
        
        st.session_state.data = {
            'time': time_steps,
            'true_values': targets,
            'predictions': predictions
        }
    
    def show_metrics(self, true_vals, pred_vals, data_count):
        mae = np.mean(np.abs(true_vals - pred_vals))
        mse = np.mean((true_vals - pred_vals) ** 2)
        rmse = np.sqrt(mse)
        error_threshold = 0.1  # TODO 设定误差阈值
        errors = np.abs(true_vals - pred_vals)
        monitoring_error_rate = np.mean(errors > error_threshold) * 100
        anomaly_detection_rate = 0.0
        if st.session_state.abnormal_regions and hasattr(st.session_state, 'data'):
            data = st.session_state.data
            if st.session_state.selected_range is not None:
                start_time, end_time = st.session_state.selected_range
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                current_time = data['time'][mask]
            else:
                current_time = data['time'][:500]
            anomaly_points = 0
            for start, end in st.session_state.abnormal_regions:
                anomaly_mask = (current_time >= start) & (current_time <= end)
                anomaly_points += np.sum(anomaly_mask)
            anomaly_detection_rate = (anomaly_points / len(current_time)) * 100
        
        st.markdown("### 📊 指标(当前范围)")
        
        metric_row1_col1, metric_row1_col2, metric_row1_col3 = st.columns(3)
        
        with metric_row1_col1:
            st.metric(
                label="平均绝对误差 (MAE)",
                value=f"{mae:.4f}",
                help="预测值与真实值的平均绝对差值"
            )
        
        with metric_row1_col2:
            st.metric(
                label="均方误差 (MSE)",
                value=f"{mse:.6f}",
                help="预测值与真实值差值的平方的平均值"
            )
        
        with metric_row1_col3:
            st.metric(
                label="均方根误差 (RMSE)",
                value=f"{rmse:.4f}",
                help="均方误差的平方根"
            )
        metric_row2_col1, metric_row2_col2, metric_row2_col3 = st.columns(3)
        
        with metric_row2_col1:
            st.metric(
                label="监测错误率",
                value=f"{monitoring_error_rate:.2f}%",
                help=f"预测误差超过{error_threshold}阈值的数据点比例"
            )
        
        with metric_row2_col2:
            st.metric(
                label="异常检测率",
                value=f"{anomaly_detection_rate:.2f}%",
                help="当前范围内被标记为异常的数据点比例"
            )
        
        with metric_row2_col3:
            st.metric(
                label="数据完整性",
                value="100%",
                help="当前数据的完整性评估"
            )
        
        st.markdown("### 📈 数据统计信息")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.info(f"**数据点数量**: {data_count}")
        
        with stat_col2:
            st.info(f"**真实值范围**: [{true_vals.min():.3f}, {true_vals.max():.3f}]")
        
        with stat_col3:
            st.info(f"**预测值范围**: [{pred_vals.min():.3f}, {pred_vals.max():.3f}]")
    
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