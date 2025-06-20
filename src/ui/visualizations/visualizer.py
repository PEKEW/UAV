import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from scipy.ndimage import label, find_objects

class Visualizer:
    """可视化工具类，负责所有的数据可视化"""
    
    @staticmethod
    def _merge_anomalous_windows(predictions, time_indices, window_size, full_time_data):
        """合并连续的异常窗口"""
        if np.sum(predictions) == 0:
            return []

        # 标记连续的异常区域
        labeled_array, num_features = label(predictions == 1)
        if num_features == 0:
            return []

        merged_regions = []
        found_objects = find_objects(labeled_array)

        for sl in found_objects:
            pred_start_index = sl[0].start
            pred_end_index = sl[0].stop - 1

            # 根据标签策略（窗口末端点），异常事件的开始时间应对应第一个异常窗口的末端
            data_start_idx = time_indices[pred_start_index] + window_size - 1
            # 异常事件的结束时间应对应最后一个异常窗口的末端
            data_end_idx = time_indices[pred_end_index] + window_size - 1

            if data_start_idx < len(full_time_data) and data_end_idx < len(full_time_data):
                start_time = full_time_data[data_start_idx]
                end_time = full_time_data[data_end_idx]
                merged_regions.append((start_time, end_time))
        
        return merged_regions

    @staticmethod
    def get_window_level_comparison(predictions, time_indices, window_size, time_data, data):
        """获取窗口级别的对比数据，与模型训练和评估一致"""
        window_predictions = []
        window_true_labels = []
        window_time_points = []
        window_correct = []
        
        for pred, start_idx in zip(predictions, time_indices):
            end_idx = start_idx + window_size - 1  # 窗口最后一个时间点
            if 'label' in data and end_idx < len(data['label']):
                # 窗口的真实标签（训练时使用的标签策略）
                true_label = data['label'][end_idx]
                
                window_predictions.append(pred)
                window_true_labels.append(true_label)
                window_time_points.append((start_idx, end_idx))
                window_correct.append(1 if pred == true_label else 0)
        
        return {
            'predictions': np.array(window_predictions),
            'true_labels': np.array(window_true_labels),
            'time_points': window_time_points,
            'correct': np.array(window_correct),
            'accuracy': np.mean(window_correct) * 100 if len(window_correct) > 0 else 0
        }
    
    @staticmethod
    def create_standard_chart(time_data, true_values, anomaly_regions=None, feature_name="特征值"):
        """创建标准视图图表"""
        fig = go.Figure()
        
        full_data = st.session_state.get('data', {})

        if 'label' in full_data and 'time' in full_data:
            # 找到当前显示数据对应的标签
            full_time_data = full_data['time']
            time_mask = (full_time_data >= time_data.min()) & (full_time_data <= time_data.max())
            current_labels = full_data['label'][time_mask]
            
            # 安全检查，确保长度一致
            if len(current_labels) == len(time_data):
                # 使用分段绘制方法，避免连接不连续的数据点
                
                # 绘制正常数据段
                diff_normal = np.diff((current_labels == 0).astype(int), prepend=0, append=0)
                starts_normal = np.where(diff_normal == 1)[0]
                ends_normal = np.where(diff_normal == -1)[0]

                for i, (start, end) in enumerate(zip(starts_normal, ends_normal)):
                    if start < end:
                        fig.add_trace(go.Scatter(
                            x=time_data[start:end],
                            y=true_values[start:end],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            name='正常数据',
                            showlegend=(i == 0),
                            hovertemplate='时间: %{x:.1f}s<br>数值: %{y:.4f}<extra></extra>'
                        ))

                # 绘制异常数据段
                diff_anomaly = np.diff((current_labels == 1).astype(int), prepend=0, append=0)
                starts_anomaly = np.where(diff_anomaly == 1)[0]
                ends_anomaly = np.where(diff_anomaly == -1)[0]
                
                for i, (start, end) in enumerate(zip(starts_anomaly, ends_anomaly)):
                    if start < end:
                        fig.add_trace(go.Scatter(
                            x=time_data[start:end],
                            y=true_values[start:end],
                            mode='lines',
                            line=dict(color='red', width=1),
                            name='异常数据',
                            showlegend=(i == 0),
                            hovertemplate='时间: %{x:.1f}s<br>数值: %{y:.4f}<extra></extra>'
                        ))
            else:
                # 长度不匹配时使用单色显示
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=true_values,
                    mode='lines',
                    name=feature_name,
                    line=dict(color='blue', width=1),
                    opacity=0.8,
                    hovertemplate='时间: %{x:.1f}s<br>数值: %{y:.4f}<extra></extra>'
                ))
        else:
            # 没有标签时使用蓝色线条
            fig.add_trace(go.Scatter(
                x=time_data,
                y=true_values,
                mode='lines',
                name=feature_name,
                line=dict(color='blue', width=1),
                opacity=0.8,
                hovertemplate='时间: %{x:.1f}s<br>数值: %{y:.4f}<extra></extra>'
            ))
        
        # 添加模型预测的异常区域（不显示数据标签）
        detection_result = st.session_state.get('detection_result', None)
        if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
            predictions = detection_result['predictions']
            time_indices = detection_result['time_indices']
            window_size = detection_result['window_size']
            full_time_data = detection_result['time_data']
            
            # 合并连续的异常窗口进行显示
            merged_anomalies = Visualizer._merge_anomalous_windows(
                predictions, time_indices, window_size, full_time_data
            )

            for start_time, end_time in merged_anomalies:
                # 只显示在当前时间范围内的预测
                if start_time >= time_data.min() and end_time <= time_data.max():
                    fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        layer="below",
                        line_width=1,
                        line_color="red"
                    )
            
            # 添加图例说明（不显示数据标签）
            anomaly_count = np.sum(predictions == 1)
            if anomaly_count > 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color='rgba(255, 0, 0, 0.3)', size=10),
                    name='模型预测异常',
                    showlegend=True
                ))
        
        
        
        # 添加性能统计信息
        detection_result = st.session_state.get('detection_result', None)
        if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
            predictions = detection_result['predictions']
            anomaly_ratio = np.mean(predictions == 1) * 100
            
            annotation_text = f"异常检测比例: {anomaly_ratio:.1f}%"
            
            # 如果有真实标签，计算准确率
            if 'label' in full_data:
                # 重新获取完整的time_indices等，确保与完整数据对齐
                time_indices = detection_result['time_indices']
                window_size = detection_result['window_size']
                
                # 修正：get_window_level_comparison也需要使用完整数据
                full_time_for_metrics = detection_result.get('time_data', full_data['time'])

                # 计算准确率
                window_comparison = Visualizer.get_window_level_comparison(
                    predictions, time_indices, window_size, full_time_for_metrics, full_data
                )
                accuracy = window_comparison['accuracy']
                annotation_text += f"<br>准确率: {accuracy:.1f}%"
            
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        # 更新布局
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
        
        return fig
    
    @staticmethod
    def create_flight_visualization(data, time_data):
        """创建飞行姿态可视化"""
        if 'redraw_flag' not in st.session_state:
            st.session_state.redraw_flag = True
            
        if not st.session_state.redraw_flag and 'last_figure' in st.session_state:
            return st.session_state.last_figure
            
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=(
                '3D轨迹', 
                '姿态角度', 
                '高度', 
                '速度'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        if data.get('has_position', False):
            if 'label' in data:
                normal_indices = data['label'] == 0
                anomaly_indices = data['label'] == 1
                
                if np.any(normal_indices):
                    fig.add_trace(
                        go.Scatter3d(
                            x=data['x'][normal_indices],
                            y=data['y'][normal_indices],
                            z=data['z'][normal_indices],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            name='真实数据-正常',
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                
                if np.any(anomaly_indices):
                    fig.add_trace(
                        go.Scatter3d(
                            x=data['x'][anomaly_indices],
                            y=data['y'][anomaly_indices],
                            z=data['z'][anomaly_indices],
                            mode='lines',
                            line=dict(color='red', width=1),
                            name='真实数据-异常',
                            showlegend=True
                        ),
                        row=1, col=1
                    )
            
            # 不显示模型预测的3D轨迹，保持简洁
            elif not 'label' in data:
                fig.add_trace(
                    go.Scatter3d(
                        x=data['x'],
                        y=data['y'],
                        z=data['z'],
                        mode='lines',
                        name='飞行轨迹',
                        line=dict(color='blue', width=1),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        if data.get('has_attitude', False):
            for angle, color in [('roll', 'green'), ('pitch', 'orange'), ('yaw', 'purple')]:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=data[angle],
                        name=angle.capitalize(),
                        line=dict(color=color, width=1),
                        opacity=0.8
                    ),
                    row=1, col=2
                )
            
            # 只在有检测结果时显示模型预测（简化版本）
            detection_result = st.session_state.get('detection_result', None)
            if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
                predictions = detection_result['predictions']
                time_indices = detection_result['time_indices']
                window_size = detection_result['window_size']
                full_time_data = detection_result['time_data']
                
                y_min = min(data['roll'].min(), data['pitch'].min(), data['yaw'].min())
                y_max = max(data['roll'].max(), data['pitch'].max(), data['yaw'].max())
                
                # 合并连续的异常窗口进行显示
                merged_anomalies = Visualizer._merge_anomalous_windows(
                    predictions, time_indices, window_size, full_time_data
                )
                
                for start_time, end_time in merged_anomalies:
                    if start_time >= time_data.min() and end_time <= time_data.max():
                        fig.add_shape(
                            type="rect",
                            x0=start_time, x1=end_time,
                            y0=y_min, y1=y_max,
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            layer="below",
                            line=dict(color="red", width=1),
                            row=1, col=2
                        )
        
        if 'z' in data:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=data['z'],
                    name='高度',
                    line=dict(color='blue', width=1),
                    opacity=0.8
                ),
                row=2, col=1
            )
            
            # 只在有检测结果时显示模型预测（简化版本）
            detection_result = st.session_state.get('detection_result', None)
            if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
                predictions = detection_result['predictions']
                time_indices = detection_result['time_indices']
                window_size = detection_result['window_size']
                full_time_data = detection_result['time_data']
                
                # 合并连续的异常窗口进行显示
                merged_anomalies = Visualizer._merge_anomalous_windows(
                    predictions, time_indices, window_size, full_time_data
                )

                for start_time, end_time in merged_anomalies:
                    if start_time >= time_data.min() and end_time <= time_data.max():
                        fig.add_shape(
                            type="rect",
                            x0=start_time, x1=end_time,
                            y0=data['z'].min(),
                            y1=data['z'].max(),
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            layer="below",
                            line=dict(color="red", width=1),
                            row=2, col=1
                        )
        
        if 'velocity' in data:
            velocity_data = data['velocity']
        elif data.get('has_position', False):
            from src.data_processing.data_processor import DataProcessor
            velocity_data = DataProcessor.calculate_velocity(time_data, data['x'], data['y'], data['z'])
        else:
            velocity_data = None
            
        if velocity_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=velocity_data,
                    name='速度',
                    line=dict(color='orange', width=1),
                    opacity=0.8
                ),
                row=2, col=2
            )
            
            # 只在有检测结果时显示模型预测（简化版本）
            detection_result = st.session_state.get('detection_result', None)
            if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
                predictions = detection_result['predictions']
                time_indices = detection_result['time_indices']
                window_size = detection_result['window_size']
                full_time_data = detection_result['time_data']
                
                # 合并连续的异常窗口进行显示
                merged_anomalies = Visualizer._merge_anomalous_windows(
                    predictions, time_indices, window_size, full_time_data
                )
                
                for start_time, end_time in merged_anomalies:
                    if start_time >= time_data.min() and end_time <= time_data.max():
                        fig.add_shape(
                            type="rect",
                            x0=start_time, x1=end_time,
                            y0=velocity_data.min(),
                            y1=velocity_data.max(),
                            fillcolor="rgba(255, 0, 0, 0.3)",
                            layer="below",
                            line=dict(color="red", width=1),
                            row=2, col=2
                        )
        
        model_detection_completed = st.session_state.get('model_detection_completed', False)

        if 'label' in data and not model_detection_completed:
            label_diff = np.diff(np.pad(data['label'], (1, 1), 'constant'))
            anomaly_starts = time_data[np.where(label_diff == 1)[0]]
            anomaly_ends = time_data[np.where(label_diff == -1)[0] - 1]
            
            for start, end in zip(anomaly_starts, anomaly_ends):
                if data.get('has_attitude', False):
                    y_min = min(data['roll'].min(), data['pitch'].min(), data['yaw'].min())
                    y_max = max(data['roll'].max(), data['pitch'].max(), data['yaw'].max())
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=y_min, y1=y_max,
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        layer="below",
                        line=dict(color="red", width=1),
                        row=1, col=2
                    )
                
                if 'z' in data:
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=data['z'].min(),
                        y1=data['z'].max(),
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        layer="below",
                        line=dict(color="red", width=1),
                        row=2, col=1
                    )
                
                if velocity_data is not None:
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=velocity_data.min(),
                        y1=velocity_data.max(),
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        layer="below",
                        line=dict(color="red", width=1),
                        row=2, col=2
                    )
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title=dict(
                text="飞行姿态异常检测分析",
                x=0.5,
                font=dict(size=18)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        fig.update_scenes(
            aspectmode='data',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor='white'
        )
        
        for row in [1, 2]:
            for col in [1, 2]:
                if row == 1 and col == 1:
                    continue
                fig.update_xaxes(
                    title_text="时间 (s)",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    row=row,
                    col=col
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    row=row,
                    col=col
                )
        
        fig.update_yaxes(title_text="角度 (度)", row=1, col=2)
        fig.update_yaxes(title_text="高度 (m)", row=2, col=1)
        fig.update_yaxes(title_text="速度 (m/s)", row=2, col=2)
        
        # 显示统计信息
        detection_result = st.session_state.get('detection_result', None)
        if detection_result and 'predictions' in detection_result and st.session_state.get('model_detection_completed', False):
            predictions = detection_result['predictions']
            time_indices = detection_result['time_indices']
            window_size = detection_result['window_size']
            anomaly_ratio = np.mean(predictions == 1) * 100
            
            full_data = st.session_state.get('data', {})
            
            # 获取准确率
            if 'label' in full_data:
                # BUGFIX: 确保使用完整数据计算指标
                full_time_data_for_metrics = detection_result.get('time_data', full_data.get('time'))
                window_comparison = Visualizer.get_window_level_comparison(
                    predictions, time_indices, window_size, full_time_data_for_metrics, full_data
                )
                accuracy = window_comparison['accuracy']
                
                fig.add_annotation(
                    text=(f"异常比例: {anomaly_ratio:.1f}%<br>" +
                          f"预测准确率: {accuracy:.1f}%"),
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)"
                )
            else:
                fig.add_annotation(
                    text=f"异常检测比例: {anomaly_ratio:.1f}%",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=11)
                )
        
        st.session_state.redraw_flag = False
        st.session_state.last_figure = fig
        return fig
    
    @staticmethod
    def trigger_redraw():
        """触发重绘"""
        st.session_state.redraw_flag = True 