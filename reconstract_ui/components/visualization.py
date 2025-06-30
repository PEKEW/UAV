"""
Visualization components for battery and flight data
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizationComponent:
    """Handles data visualization and anomaly detection display"""
    
    def render_chart_area(self):
        """Render the main chart area"""
        st.markdown("##  数据可视化与异常检测")
        
        if st.session_state.data_loaded and st.session_state.data is not None:
            data = st.session_state.data
            data_type = getattr(st.session_state, 'data_type', 'unknown')
            
            self._show_data_info(data)
            self._render_anomaly_label_control()
            self._render_time_range_selector(data)
            
            filtered_data = self._filter_data_by_time_range(data)
            
            if data_type == 'battery':
                self._create_battery_visualization(filtered_data)
            elif data_type == 'flight':
                self._create_flight_visualization(filtered_data)
            else:
                st.warning("未知数据类型，无法创建可视化")
        else:
            st.info("请先上传数据文件开始分析")
    
    def _show_data_info(self, data):
        """Display basic data information"""
        total_points = len(data['time'])
        time_min, time_max = data['time'].min(), data['time'].max()
        total_duration = time_max - time_min
        st.info(f"数据总量: {total_points} 个数据点 | 时间范围: {time_min:.1f}s - {time_max:.1f}s | 总时长: {total_duration:.1f}s")
    
    def _render_anomaly_label_control(self):
        """Render anomaly label control"""
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
                data = st.session_state.data
                has_labels = self._check_data_has_labels(data)
                if has_labels:
                    st.success("✓ 数据中存在异常标签")
                else:
                    st.warning("⚠ 数据中未找到异常标签")
        
        st.markdown("---")
    
    def _check_data_has_labels(self, data):
        """Check if data contains anomaly labels"""
        return 'data_labels' in data and len(data['data_labels']) > 0
    
    def _render_time_range_selector(self, data):
        """Render time range selector"""
        st.markdown("### 时间范围选择")
        
        time_min, time_max = data['time'].min(), data['time'].max()
        total_duration = time_max - time_min
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_options = np.arange(time_min, time_max, 30)
            start_time = st.selectbox(
                "起始时间 (秒)",
                options=start_options,
                format_func=lambda x: f"{x:.0f}s",
                key="start_time_selector"
            )
        
        with col2:
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
        
        # Display current range
        if st.session_state.selected_time_range:
            start, end = st.session_state.selected_time_range
            duration = end - start
            st.info(f"当前显示范围: {start:.0f}s - {end:.0f}s (时长: {duration:.0f}s)")
        else:
            st.info(f"当前显示全部数据 (总时长: {total_duration:.0f}s)")
        
        st.markdown("---")
    
    def _filter_data_by_time_range(self, data):
        """Filter data by selected time range"""
        if st.session_state.selected_time_range is None:
            return data
        
        start_time, end_time = st.session_state.selected_time_range
        time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
        
        filtered_data = {
            'time': data['time'][time_mask],
            'features': {}
        }
        
        # Filter features
        for feature_name, feature_data in data['features'].items():
            filtered_data['features'][feature_name] = feature_data[time_mask]
        
        # Filter anomaly regions
        if 'anomaly_regions' in data:
            filtered_anomaly_regions = []
            for region in data['anomaly_regions']:
                region_start, region_end = region['start'], region['end']
                if region_start <= end_time and region_end >= start_time:
                    adjusted_region = region.copy()
                    adjusted_region['start'] = max(region_start, start_time)
                    adjusted_region['end'] = min(region_end, end_time)
                    filtered_anomaly_regions.append(adjusted_region)
            filtered_data['anomaly_regions'] = filtered_anomaly_regions
        
        # Filter data labels
        if 'data_labels' in data:
            filtered_data_labels = []
            for label in data['data_labels']:
                label_start, label_end = label['start'], label['end']
                if label_start <= end_time and label_end >= start_time:
                    adjusted_label = label.copy()
                    adjusted_label['start'] = max(label_start, start_time)
                    adjusted_label['end'] = min(label_end, end_time)
                    filtered_data_labels.append(adjusted_label)
            filtered_data['data_labels'] = filtered_data_labels
        
        return filtered_data
    
    def _create_battery_visualization(self, data):
        """Create battery data visualization"""
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
        
        colors = {
            'Ecell_V': 'blue', 'I_mA': 'red', 'EnergyCharge_W_h': 'red',
            'EnergyDischarge_W_h': 'orange', 'QCharge_mA_h': 'purple',
            'QDischarge_mA_h': 'brown', 'Temperature__C': 'magenta'
        }
        
        # Add feature traces
        feature_positions = [
            ('Ecell_V', 1, 1, '电池电压'),
            ('I_mA', 1, 2, '电流'),
            ('EnergyCharge_W_h', 2, 1, '充电能量'),
            ('EnergyDischarge_W_h', 2, 2, '放电能量'),
            ('Temperature__C', 3, 1, '温度')
        ]
        
        for feature, row, col, name in feature_positions:
            if feature in features:
                fig.add_trace(go.Scatter(
                    x=time_data, y=features[feature],
                    mode='lines', name=name,
                    line=dict(color=colors[feature], width=1)
                ), row=row, col=col)
        
        # Add capacity traces
        if 'QCharge_mA_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['QCharge_mA_h'],
                mode='lines', name='充电容量',
                line=dict(color=colors['QCharge_mA_h'], width=1)
            ), row=3, col=2)
        
        if 'QDischarge_mA_h' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['QDischarge_mA_h'],
                mode='lines', name='放电容量',
                line=dict(color=colors['QDischarge_mA_h'], width=1)
            ), row=3, col=2)
        
        # Add anomaly regions
        self._add_anomaly_regions_to_battery_chart(fig, data, features)
        
        fig.update_layout(title='电池数据异常检测', height=1000, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key="battery_chart")
        
        self._show_battery_metrics(data)
    
    def _create_flight_visualization(self, data):
        """Create flight data visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('姿态角 (Roll, Pitch, Yaw)', '速度', '3D轨迹', '位置 (X, Y, Z)'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter3d", "secondary_y": False}, {"type": "scatter"}]],
            vertical_spacing=0.12
        )
        
        time_data = data['time']
        features = data['features']
        
        subplot_has_data = {(1, 1): False, (1, 2): False, (2, 1): False, (2, 2): False}
        
        # Add attitude traces
        attitude_colors = {'roll': 'red', 'pitch': 'blue', 'yaw': 'orange'}
        for attitude in ['roll', 'pitch', 'yaw']:
            if attitude in features:
                fig.add_trace(go.Scatter(
                    x=time_data, y=features[attitude],
                    mode='lines', name=attitude.capitalize(),
                    line=dict(color=attitude_colors[attitude], width=1)
                ), row=1, col=1)
                subplot_has_data[(1, 1)] = True
        
        # Add velocity trace
        if 'velocity' in features:
            fig.add_trace(go.Scatter(
                x=time_data, y=features['velocity'],
                mode='lines', name='速度',
                line=dict(color='purple', width=1)
            ), row=1, col=2)
            subplot_has_data[(1, 2)] = True
        
        # Add 3D trajectory
        self._add_3d_trajectory(fig, data, features, subplot_has_data)
        
        # Add position traces
        pos_colors = {'x': 'red', 'y': 'red', 'z': 'blue'}
        for pos in ['x', 'y', 'z']:
            if pos in features:
                fig.add_trace(go.Scatter(
                    x=time_data, y=features[pos],
                    mode='lines', name=f'{pos.upper()}位置',
                    line=dict(color=pos_colors[pos], width=1)
                ), row=2, col=2)
                subplot_has_data[(2, 2)] = True
        
        # Add anomaly regions
        self._add_anomaly_regions_to_flight_chart(fig, data, features, subplot_has_data)
        
        fig.update_layout(title='飞行数据异常检测', height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key="flight_chart")
        
        self._show_flight_metrics(data)
    
    def _add_3d_trajectory(self, fig, data, features, subplot_has_data):
        """Add 3D trajectory to flight visualization"""
        if not all(coord in features for coord in ['x', 'y', 'z']):
            return
        
        time_data = data['time']
        
        if 'original_samples' in data:
            # H5 data: separate samples
            original_samples = data['original_samples']
            sample_info = data['sample_info']
            n_samples = sample_info['n_samples']
            feature_names = sample_info['feature_names']
            
            coord_indices = {
                coord: feature_names.index(coord) if coord in feature_names else None
                for coord in ['x', 'y', 'z']
            }
            
            if all(idx is not None for idx in coord_indices.values()):
                for i in range(n_samples):
                    sample_coords = {
                        coord: original_samples[i, :, idx]
                        for coord, idx in coord_indices.items()
                    }
                    sample_time = np.arange(i * 30, (i + 1) * 30)
                    
                    is_anomaly = False
                    if 'original_labels' in data and i < len(data['original_labels']):
                        is_anomaly = data['original_labels'][i] == 1
                    
                    color = 'rgba(255, 100, 100, 0.8)' if is_anomaly else 'rgba(70, 130, 180, 0.8)'
                    
                    fig.add_trace(go.Scatter3d(
                        x=sample_coords['x'], y=sample_coords['y'], z=sample_coords['z'],
                        mode='markers', marker=dict(size=1, color=color),
                        name='3D轨迹' if i == 0 else None, showlegend=(i == 0),
                        hovertemplate=f'样本{i+1}<br>时间: %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
                        text=[f'{t:.1f}s' for t in sample_time]
                    ), row=2, col=1)
        else:
            # CSV data: continuous trajectory
            fig.add_trace(go.Scatter3d(
                x=features['x'], y=features['y'], z=features['z'],
                mode='lines', line=dict(color='rgba(70, 130, 180, 0.8)', width=4),
                name='3D飞行轨迹', showlegend=True,
                hovertemplate='时间: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                text=[f'{t:.1f}s' for t in time_data]
            ), row=2, col=1)
        
        subplot_has_data[(2, 1)] = True
    
    def _add_anomaly_regions_to_battery_chart(self, fig, data, features):
        """Add anomaly regions to battery chart"""
        # Model detected anomalies (red)
        if st.session_state.model_detection_completed and 'anomaly_regions' in data:
            self._add_anomaly_regions(fig, data['anomaly_regions'], features, 'model', 
                                    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)])
        
        # Data label anomalies (orange)
        if st.session_state.show_data_labels and 'data_labels' in data:
            self._add_anomaly_regions(fig, data['data_labels'], features, 'data',
                                    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)])
    
    def _add_anomaly_regions_to_flight_chart(self, fig, data, features, subplot_has_data):
        """Add anomaly regions to flight chart"""
        valid_positions = [(row, col) for (row, col) in [(1, 1), (1, 2), (2, 2)] 
                          if subplot_has_data[(row, col)]]
        
        # Model detected anomalies (red)
        if st.session_state.model_detection_completed and 'anomaly_regions' in data:
            self._add_anomaly_regions(fig, data['anomaly_regions'], features, 'model', valid_positions)
        
        # Data label anomalies (orange)
        if st.session_state.show_data_labels and 'data_labels' in data:
            self._add_anomaly_regions(fig, data['data_labels'], features, 'data', valid_positions)
    
    def _add_anomaly_regions(self, fig, regions, features, region_type, positions):
        """Add anomaly regions to chart"""
        color = "rgba(255, 20, 20, {alpha})" if region_type == 'model' else "rgba(255, 165, 0, 0.25)"
        name = "模型检测异常" if region_type == 'model' else "数据标签异常"
        
        for i, region in enumerate(regions):
            start, end = region['start'], region['end']
            confidence = region.get('confidence', 0.5) if region_type == 'model' else None
            anomaly_ratio = region.get('anomaly_ratio', 0.5) if region_type == 'data' else None
            
            for row, col in positions:
                y_data = self._get_subplot_data(row, col, features)
                if y_data is None:
                    continue
                
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
                
                if region_type == 'model':
                    alpha = 0.4 + 0.4 * confidence
                    fill_color = color.format(alpha=alpha)
                    hover_text = f"{name}<br>时间: {start:.1f}s-{end:.1f}s<br>置信度: {confidence:.3f}"
                else:
                    fill_color = color
                    hover_text = f"{name}<br>时间: {start:.1f}s-{end:.1f}s<br>异常比例: {anomaly_ratio:.3f}"
                
                fig.add_trace(go.Scatter(
                    x=[start, start, end, end, start],
                    y=[y_min, y_max, y_max, y_min, y_min],
                    fill="toself", fillcolor=fill_color, line=dict(width=0), mode="none",
                    showlegend=(i == 0), name=name if i == 0 else None,
                    hovertemplate=f"{hover_text}<extra></extra>"
                ), row=row, col=col)
    
    def _get_subplot_data(self, row, col, features):
        """Get data for specific subplot position"""
        feature_map = {
            (1, 1): ['Ecell_V', ['roll', 'pitch', 'yaw']],
            (1, 2): ['I_mA', 'velocity'],
            (2, 1): ['EnergyCharge_W_h', None],
            (2, 2): ['EnergyDischarge_W_h', ['x', 'y', 'z']],
            (3, 1): ['Temperature__C', None],
            (3, 2): [['QCharge_mA_h', 'QDischarge_mA_h'], None]
        }
        
        expected_features = feature_map.get((row, col))
        if not expected_features:
            return None
        
        # Handle battery features
        if isinstance(expected_features[0], str) and expected_features[0] in features:
            return features[expected_features[0]]
        elif isinstance(expected_features[0], list):
            available_features = [f for f in expected_features[0] if f in features]
            if available_features:
                return np.concatenate([features[f] for f in available_features])
        
        # Handle flight features
        if len(expected_features) > 1 and expected_features[1]:
            if isinstance(expected_features[1], str) and expected_features[1] in features:
                return features[expected_features[1]]
            elif isinstance(expected_features[1], list):
                available_features = [f for f in expected_features[1] if f in features]
                if available_features:
                    return np.concatenate([features[f] for f in available_features])
        
        return None
    
    def _show_battery_metrics(self, data):
        """Show battery data metrics"""
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
    
    def _show_flight_metrics(self, data):
        """Show flight data metrics"""
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