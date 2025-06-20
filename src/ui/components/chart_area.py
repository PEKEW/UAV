import streamlit as st
import numpy as np
from .base_component import BaseComponent
from src.ui.visualizations.visualizer import Visualizer

class ChartArea(BaseComponent):
    """图表区域组件"""
    
    def init_session_state(self):
        """初始化session state"""
        if 'selected_range' not in st.session_state:
            st.session_state.selected_range = None
        if 'model_detection_completed' not in st.session_state:
            st.session_state.model_detection_completed = False
    
    def render(self):
        """渲染图表区域"""
        st.markdown("##  数据可视化与异常检测")
        
        if not self.get_session_state('data_loaded') or self.get_session_state('data') is None:
            st.info("请先导入数据")
            return
        
        data = self.get_session_state('data')
        self._render_visualization_mode(data)
    
    def _render_visualization_mode(self, data):
        """渲染可视化模式选择和图表"""
        # 根据数据类型自动选择默认视图
        has_flight_data = data.get('has_attitude', False) or data.get('has_position', False)
        has_battery_features = any(feature in data for feature in ['Ecell_V', 'I_mA', 'Temperature__C'])
        
        # 自动选择默认模式
        if has_flight_data and not has_battery_features:
            default_mode = "飞行姿态视图"
        elif has_battery_features and not has_flight_data:
            default_mode = "标准视图"
        else:
            # 两种数据都有或都没有，默认标准视图
            default_mode = "标准视图"
        
        # 获取默认索引
        options = ["标准视图", "飞行姿态视图"]
        default_index = options.index(default_mode)
        
        viz_mode = st.radio(
            "可视化模式 (已自动选择适合的模式)",
            options,
            index=default_index,
            horizontal=True,
            help=f"根据数据类型自动选择了{default_mode}"
        )
        
        self._show_data_info(data)
        
        if viz_mode == "标准视图":
            self._render_standard_view(data)
        else:
            self._render_flight_view(data)
    
    def _show_data_info(self, data):
        """显示数据信息"""
        total_points = len(data['time'])
        time_min, time_max = data['time'].min(), data['time'].max()
        total_duration = time_max - time_min
        
        st.info(f"数据总量: {total_points} 个数据点 | 时间范围: {time_min:.1f}s - {time_max:.1f}s | 总时长: {total_duration:.1f}s")
    
    def _render_standard_view(self, data):
        """渲染标准视图"""
        # 使用可视化配置的最大时长，如果没有配置则使用默认值
        max_viz_duration = self.get_session_state('viz_max_duration', 300)  # 默认5分钟
        time_data, display_data = self._get_display_data(data, max_viz_duration)
        
        if len(time_data) == 0:
            st.warning("选择的时间范围内没有数据")
            return
        
        # 创建图表
        fig = Visualizer.create_standard_chart(
            time_data=time_data,
            true_values=display_data['true_values'],
            anomaly_regions=display_data.get('anomaly_regions'),
            feature_name=self.get_session_state('selected_feature_name', "特征值")
        )
        
        st.plotly_chart(fig, use_container_width=True, key="main_chart")
    
    def _render_flight_view(self, data):
        """渲染飞行姿态视图"""
        if not (data.get('has_attitude', False) or data.get('has_position', False)):
            st.warning("数据中缺少飞行姿态或位置信息，无法创建飞行可视化")
            return
        
        # 为飞行视图添加数据选择控制，使用可视化配置的最大时长
        max_viz_duration = self.get_session_state('viz_max_duration', 300)  # 默认5分钟
        time_data, display_data = self._get_flight_display_data(data, max_viz_duration)
        
        if len(time_data) == 0:
            st.warning("选择的时间范围内没有数据")
            return
        
        fig = Visualizer.create_flight_visualization(display_data, time_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_display_data(self, data, max_range):
        """获取要显示的数据范围，返回切片后的时间和完整数据字典"""
        display_data = {}

        if self.get_session_state('selected_range') is not None:
            start_time, end_time = self.get_session_state('selected_range')
            
            selected_duration = end_time - start_time
            if selected_duration > max_range:
                end_time = start_time + max_range
                st.warning(f"选择的时间范围超过{max_range}秒限制，已自动截取到 {start_time:.1f}s - {end_time:.1f}s")
            
            mask = (data['time'] >= start_time) & (data['time'] <= end_time)
            time_data = data['time'][mask]

            # 像飞行视图一样，切片所有相关数组
            for key, value in data.items():
                if isinstance(value, np.ndarray) and len(value) == len(data['time']):
                    display_data[key] = value[mask]
                else:
                    display_data[key] = value # 保持非数组或长度不匹配的值
            
            actual_duration = end_time - start_time
            range_points = len(time_data)
            
            st.info(f"显示范围: {start_time:.1f}s - {end_time:.1f}s (时长: {actual_duration:.1f}s, {range_points} 个点)")
        else:
            # 使用可视化配置的默认范围
            viz_default_range = self.get_session_state('viz_default_range')
            time_min = data['time'].min()
            time_max = data['time'].max()
            total_duration = time_max - time_min
            
            if viz_default_range:
                start_time, end_time = viz_default_range
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                time_data = data['time'][mask]
                
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and len(value) == len(data['time']):
                        display_data[key] = value[mask]
                    else:
                        display_data[key] = value
                
                actual_duration = end_time - start_time
                actual_points = len(time_data)
                st.info(f"显示默认范围: {start_time:.1f}s - {end_time:.1f}s (时长: {actual_duration:.1f}s, {actual_points} 个点) | 总时长: {total_duration:.1f}s")
            elif total_duration > max_range:
                max_time = time_min + max_range
                mask = data['time'] <= max_time
                time_data = data['time'][mask]

                for key, value in data.items():
                    if isinstance(value, np.ndarray) and len(value) == len(data['time']):
                        display_data[key] = value[mask]
                    else:
                        display_data[key] = value

                actual_points = len(time_data)
                st.info(f"显示: 前{max_range}秒 ({time_min:.1f}s - {max_time:.1f}s, {actual_points} 个点) | 总时长: {total_duration:.1f}s")
            else:
                time_data = data['time']
                display_data = data.copy()
                st.info(f"显示全部: {total_duration:.1f}s ({len(time_data)} 个点)")
        
        # 确保 display_data 包含 'time' 和 'true_values'
        if 'time' not in display_data:
            display_data['time'] = time_data
        if 'true_values' not in display_data and 'true_values' in data:
            if len(display_data['time']) == len(data['true_values']):
                 display_data['true_values'] = data['true_values']
            else: # 需要mask
                 mask = (data['time'] >= time_data.min()) & (data['time'] <= time_data.max())
                 display_data['true_values'] = data['true_values'][mask]


        return time_data, display_data
    
    def _get_flight_display_data(self, data, max_range):
        """获取飞行数据要显示的数据范围"""
        if self.get_session_state('selected_range') is not None:
            start_time, end_time = self.get_session_state('selected_range')
            
            selected_duration = end_time - start_time
            if selected_duration > max_range:
                end_time = start_time + max_range
                st.warning(f"选择的时间范围超过{max_range}秒限制，已自动截取到 {start_time:.1f}s - {end_time:.1f}s")
            
            mask = (data['time'] >= start_time) & (data['time'] <= end_time)
            time_data = data['time'][mask]
            
            # 创建显示数据字典
            display_data = {'time': time_data}
            for key, value in data.items():
                if key != 'time' and isinstance(value, np.ndarray) and len(value) == len(data['time']):
                    display_data[key] = value[mask]
                else:
                    display_data[key] = value
            
            actual_duration = end_time - start_time
            range_points = len(time_data)
            
            st.info(f"显示范围: {start_time:.1f}s - {end_time:.1f}s (时长: {actual_duration:.1f}s, {range_points} 个点)")
        else:
            # 使用可视化配置的默认范围
            viz_default_range = self.get_session_state('viz_default_range')
            time_min = data['time'].min()
            time_max = data['time'].max()
            total_duration = time_max - time_min
            
            if viz_default_range:
                # 使用配置的默认范围
                start_time, end_time = viz_default_range
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                time_data = data['time'][mask]
                
                # 创建显示数据字典
                display_data = {'time': time_data}
                for key, value in data.items():
                    if key != 'time' and isinstance(value, np.ndarray) and len(value) == len(data['time']):
                        display_data[key] = value[mask]
                    else:
                        display_data[key] = value
                
                actual_duration = end_time - start_time
                actual_points = len(time_data)
                st.info(f"飞行数据默认范围: {start_time:.1f}s - {end_time:.1f}s (时长: {actual_duration:.1f}s, {actual_points} 个点) | 总时长: {total_duration:.1f}s")
            elif total_duration > max_range:
                # 回退到前N秒的逻辑
                max_time = time_min + max_range
                mask = data['time'] <= max_time
                time_data = data['time'][mask]
                
                # 创建显示数据字典
                display_data = {'time': time_data}
                for key, value in data.items():
                    if key != 'time' and isinstance(value, np.ndarray) and len(value) == len(data['time']):
                        display_data[key] = value[mask]
                    else:
                        display_data[key] = value
                
                actual_points = len(time_data)
                st.info(f"飞行数据显示: 前{max_range}秒 ({time_min:.1f}s - {max_time:.1f}s, {actual_points} 个点) | 总时长: {total_duration:.1f}s")
            else:
                # 显示全部数据
                time_data = data['time']
                display_data = data.copy()
                st.info(f"飞行数据显示全部: {total_duration:.1f}s ({len(time_data)} 个点)")
        
        return time_data, display_data
    
    
 