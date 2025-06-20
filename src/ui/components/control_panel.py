import streamlit as st
import time
from pathlib import Path
from .base_component import BaseComponent
from src.data_processing.data_processor import DataProcessor
from src.ui.visualizations.visualizer import Visualizer
import numpy as np
import io
import torch
from src.models.model import BatteryAnomalyLSTM, FlightAnomalyLSTM

class ControlPanel(BaseComponent):
    
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
    
    def render(self):
        with st.container():
            st.markdown('<div class="control-panel">控制面板</div>', unsafe_allow_html=True)
            
            self._render_monitor_mode()
            st.markdown("---")
            self._render_data_selection()
            st.markdown("---")
            self._render_model_selection()
    
    def _render_monitor_mode(self):
        st.markdown("### 监测模式 ")
        col1, col2 = st.columns(2)
        
        with col1:
            # MARK 这里是是实时监测的接口
            if st.button("实时监测", use_container_width=True):
                st.toast("️不行！", icon="⚠️")
                self.set_session_state('current_status', 'offline')
        
        with col2:
            if st.button("离线监测", use_container_width=True):
                self.set_session_state('current_status', 'offline')
        
        if self.get_session_state('current_status') == 'online':
            pass
            # st.markdown('<div class="status-online"> 没有实时监测！</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline">离线监测</div>', unsafe_allow_html=True)
    
    # TODO 在线监测这里没有
    def _render_data_selection(self):
        st.markdown("### 数据选择")
        uploaded_file = st.file_uploader(
            "点击导入或拖拽数据文件到此处 - 要求是电池数据或飞行姿态数据",
            type=['csv', 'xlsx'],
            help="支持CSV和Excel格式"
        )
        
        if st.button("导入数据", use_container_width=True, 
                    disabled=not (uploaded_file is not None and 
                                self.get_session_state('current_status') == 'offline')):
            self._load_data(uploaded_file)
        
        if self.get_session_state('data_loaded'):
            st.success("数据已导入")
            # 只有在数据已加载时才显示可视化配置
            data = self.get_session_state('data')
            if data is not None:
                self._configure_visualization_range(data)
        else:
            st.info("未导入数据")
    
    def _render_model_selection(self):
        """渲染模型选择部分"""
        st.markdown("### 异常识别")
        
        st.markdown("#### 模型选择")
        uploaded_model = st.file_uploader(
            "点击导入或拖拽模型文件到此处 - 要求是对应模型封装",
            type=['pkl', 'joblib', 'h5', 'pth', 'pt', 'onnx'],
            help="支持pkl, joblib, h5, pth, pt, onnx格式",
            key="model_uploader"
        )
        
        if st.button("加载模型", use_container_width=True, 
                    disabled=not (uploaded_model is not None and 
                                self.get_session_state('current_status') == 'offline')):
            self._load_model(uploaded_model)
        
        if self.get_session_state('model_loaded'):
            st.success("模型已加载")
            model_info = self.get_session_state('model_info')
            if model_info:
                with st.expander("模型信息"):
                    for key, value in model_info.items():
                        st.write(f"**{key}**: {value}")
        else:
            st.info("未加载模型")
        
        if st.button("开始识别(务必确保模型类型和识别特征匹配)", use_container_width=True, 
                     disabled=not (self.get_session_state('data_loaded') and 
                                 self.get_session_state('model_loaded'))):
            self._start_detection()
    
    def _load_data(self, uploaded_file):
        """加载数据"""
        with st.spinner('加载数据...'):
            try:
                df = DataProcessor.load_data(uploaded_file)
                st.success(f"成功读取数据，形状: {df.shape}")
                
                with st.expander("数据预览"):
                    st.dataframe(df.head())
                
                column_mapping, available_features = DataProcessor.detect_columns(df)
                if not column_mapping:
                    st.error("请选择要分析的数据特征")
                    return
                
                processed_data = DataProcessor.process_data(df, column_mapping)
                if processed_data is None:
                    return
                
                self.set_session_state('data', processed_data)
                self.set_session_state('data_loaded', True)
                self.set_session_state('available_features', available_features)
                self.set_session_state('original_df', df)
                
                # 设置默认特征
                if available_features:
                    self.set_session_state('selected_feature_name', available_features[0][2])
                    self.set_session_state('current_feature', available_features[0])
                
                # 重置其他状态
                self.set_session_state('selected_range', None)
                self.set_session_state('model_detection_completed', False)
                
                # 清除模型检测结果
                if 'detection_result' in st.session_state:
                    del st.session_state.detection_result
                if 'last_figure' in st.session_state:
                    del st.session_state.last_figure
                
                if hasattr(st.session_state, 'preview_start'):
                    delattr(st.session_state, 'preview_start')
                if hasattr(st.session_state, 'preview_end'):
                    delattr(st.session_state, 'preview_end')
                
                # 清除可视化配置以允许重新配置
                if 'viz_max_duration' in st.session_state:
                    del st.session_state.viz_max_duration
                if 'viz_start_position' in st.session_state:
                    del st.session_state.viz_start_position
                if 'viz_default_range' in st.session_state:
                    del st.session_state.viz_default_range
                
                st.info("已清除之前的选择范围和检测结果")
                
                # 触发重绘
                Visualizer.trigger_redraw()
                    
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
                self.set_session_state('data_loaded', False)
                self.set_session_state('data', None)
    
    def _configure_visualization_range(self, processed_data):
        """配置可视化时间范围"""
        if 'time' not in processed_data:
            return
            
        time_data = processed_data['time']
        total_duration = time_data.max() - time_data.min()
        total_points = len(time_data)
        
        st.markdown("### 可视化配置")
        st.info(f"数据总时长: {total_duration:.1f}秒 ({total_points} 个数据点)")
        
        # 获取或设置默认的可视化时长
        current_viz_duration = self.get_session_state('viz_max_duration')
        if current_viz_duration is None:
            # 首次加载时设置默认值
            default_duration = min(300, int(total_duration))
        else:
            # 使用已保存的值
            default_duration = current_viz_duration
        
        # 可视化配置：时长滑动条 + 起始位置输入框
        # 使用数据的hash作为key的一部分，确保新数据时slider重置
        data_hash = hash(str(total_duration) + str(total_points))
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_viz_duration = st.slider(
                "可视化时长 (秒)",
                min_value=60,    # 最少1分钟
                max_value=600,   # 最多10分钟
                value=default_duration,
                step=30,
                key=f"viz_duration_slider_{data_hash}",  # 动态key确保新数据时重置
                help="选择可视化的时间长度"
            )
        
        with col2:
            # 获取或设置默认的起始位置
            current_start_pos = self.get_session_state('viz_start_position')
            if current_start_pos is None:
                default_start = float(time_data.min())
            else:
                default_start = max(float(time_data.min()), min(current_start_pos, float(time_data.max() - max_viz_duration)))
            
            viz_start_pos = st.number_input(
                "起始时间 (秒)",
                min_value=float(time_data.min()),
                max_value=float(time_data.max() - max_viz_duration),
                value=default_start,
                step=10.0,
                key=f"viz_start_input_{data_hash}",
                help="选择可视化的起始时间点",
                format="%.1f"
            )
        
        # 计算显示范围
        viz_start = viz_start_pos
        viz_end = min(viz_start + max_viz_duration, time_data.max())
        
        # 保存可视化配置
        self.set_session_state('viz_max_duration', max_viz_duration)
        self.set_session_state('viz_start_position', viz_start_pos)
        self.set_session_state('viz_default_range', (viz_start, viz_end))
        
        actual_duration = viz_end - viz_start
        actual_points = len(time_data[(time_data >= viz_start) & (time_data <= viz_end)])
        
        if total_duration > max_viz_duration:
            st.warning(f"数据总时长 ({total_duration:.1f}s) 超过可视化范围 ({max_viz_duration}s)，可在图表区域手动选择其他时间段")
    
    def _load_model(self, uploaded_model):
        with st.spinner('加载模型...'):
            try:
                file_extension = Path(uploaded_model.name).suffix.lower()
                filename = uploaded_model.name.lower()
                
                # 检查文件名格式
                if not (filename.startswith('battery.') or filename.startswith('flight.')):
                    raise ValueError("模型文件名格式错误！必须以'battery.'或'flight.'开头")
                
                model_info = {
                    "filename": uploaded_model.name,
                    "filesize": f"{uploaded_model.size / 1024:.2f} KB",
                    "filetype": file_extension,
                }
                
                # 判断模型类型前缀
                if filename.startswith('battery.'):
                    model_info["model_type"] = "电池预测模型"
                    model_info["model_name"] = "battery"
                elif filename.startswith('flight.'):
                    model_info["model_type"] = "飞行轨迹预测模型"
                    model_info["model_name"] = "flight"
                
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
                
                self.set_session_state('model', {
                    'name': uploaded_model.name,
                    'type': file_extension,
                    'data': uploaded_model.getvalue()
                })
                self.set_session_state('model_info', model_info)
                self.set_session_state('model_loaded', True)
                
                st.success(f"模型 {uploaded_model.name} ({model_info['model_type']}) 加载成功！")
                
                # 预加载模型用于后续检测
                # 加载模型状态字典
                checkpoint = torch.load(io.BytesIO(uploaded_model.getvalue()))
                
                # 根据模型类型创建相应的模型实例（参数必须与训练时一致）
                if model_info["model_name"] == "battery":
                    model = BatteryAnomalyLSTM(
                        input_size=7,  # 电池特征数量
                        hidden_size=128,
                        num_layers=2,
                        num_classes=2,
                        dropout=0.2,
                        use_uncertainty=True
                    )
                else:
                    model = FlightAnomalyLSTM(
                        input_size=9,  # 飞行特征数量（包括计算的acceleration）
                        hidden_size=128,
                        num_layers=2,
                        num_classes=2,
                        dropout=0.2,
                        use_uncertainty=True
                    )
                
                # 加载模型状态
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.set_session_state('torch_model', model)
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"加载模型时发生错误：{str(e)}")
    
    def _start_detection(self):
        with st.spinner('正在执行模型识别...'):
            try:
                # 获取模型和输入数据
                model = self.get_session_state('torch_model')
                input_data = self.get_session_state('data')
                model_info = self.get_session_state('model_info')
                
                if model is None or input_data is None:
                    st.error('请确保已上传模型和输入数据')
                    return
                
                window_size = 50
                stride = 1
                

                data_features = set(input_data.keys())
                
                battery_features = {'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h', 'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C'}
                flight_features = {'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'altitude'}
                
                battery_match = len(battery_features.intersection(data_features))
                flight_match = len(flight_features.intersection(data_features))
                
                detected_data_type = "battery" if battery_match > flight_match else "flight"
                
                
                # 检查模型和数据兼容性
                if detected_data_type != model_info['model_name']:
                    st.error(f"数据类型不匹配！")
                    st.error(f"检测到的数据是{detected_data_type}数据，但加载的是{model_info['model_name']}模型")
                    st.error(f"请上传正确的{model_info['model_name']}模型文件，或使用{detected_data_type}数据文件")
                    return
                
                # 根据检测到的数据类型选择特征
                if detected_data_type == "battery":
                    required_features = [
                        'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                        'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C'
                    ]
                else:
                    required_features = [
                        'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity'
                    ]
                
                
                features_list = []
                missing_features = []
                for feature in required_features:
                    if feature in input_data:
                        features_list.append(input_data[feature])
                    else:
                        missing_features.append(feature)
                        features_list.append(np.zeros_like(input_data['time']))
                
                if missing_features:
                    st.warning(f"缺失特征: {missing_features}")
                
                if detected_data_type == "flight":
                    if 'velocity' in input_data:
                        velocity = input_data['velocity']
                        if 'time' in input_data:
                            time_diff = np.diff(input_data['time'])
                            time_diff = np.where(time_diff == 0, 1e-6, time_diff)
                            velocity_diff = np.diff(velocity)
                            acceleration = velocity_diff / time_diff
                            acceleration = np.append(acceleration, acceleration[-1])  # 填充最后一个值
                        else:
                            acceleration = np.gradient(velocity)
                        features_list.append(acceleration)
                    else:
                        features_list.append(np.zeros_like(input_data['time']))
                    
                    if 'altitude' in input_data:
                        features_list.append(input_data['altitude'])
                    else:
                        features_list.append(np.zeros_like(input_data['time']))
                    
                    required_features = required_features + ['acceleration', 'altitude']
                
                features_array = np.stack(features_list, axis=1)  # [time_steps, n_features]
                
                
                if detected_data_type == "flight":
                    for i, feature in enumerate(required_features):
                        if i < features_array.shape[1]:
                            feature_data = features_array[:, i]
                        else:
                            st.write(f"  {i+1}. {feature}: **缺失！**")
                
                total_windows = len(features_array) - window_size + 1
                
                windows = []
                time_indices = []
                
                for i in range(total_windows):
                    start_idx = i * stride
                    end_idx = start_idx + window_size
                    windows.append(features_array[start_idx:end_idx])
                    time_indices.append(start_idx)
                
                windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
                
                all_predictions = []
                all_probabilities = []
                all_confidence = []
                all_uncertainties = []
                
                with torch.no_grad():
                    for window in windows_tensor:
                        window = window.unsqueeze(0)
                        confidence_results = model.predict_with_confidence(window)
                        
                        all_predictions.append(confidence_results['predictions'].cpu().numpy())
                        all_probabilities.append(confidence_results['probabilities'].cpu().numpy())
                        all_confidence.append(confidence_results['confidence'].cpu().numpy())
                        
                        if confidence_results['uncertainty'] is not None:
                            all_uncertainties.append(confidence_results['uncertainty'].cpu().numpy())
                
                all_predictions = np.concatenate(all_predictions)
                all_probabilities = np.concatenate(all_probabilities)
                all_confidence = np.concatenate(all_confidence)
                has_uncertainty = len(all_uncertainties) > 0
                if has_uncertainty:
                    all_uncertainties = np.concatenate(all_uncertainties)
                
                
                


                result = {
                    'predictions': all_predictions,
                    'probabilities': all_probabilities,
                    'confidence': all_confidence,
                    'time_indices': time_indices,
                    'uncertainties': all_uncertainties if has_uncertainty else None,
                    'window_size': window_size,
                    'stride': stride,
                    'time_data': input_data['time'],  # 添加时间数据用于可视化
                    'feature_data': features_array     # 添加特征数据用于可视化
                }
                self.set_session_state('detection_result', result)
                self.set_session_state('model_detection_completed', True)
                
                
                
                # 在检测完成后触发重绘
                Visualizer.trigger_redraw()
                

                    
            except Exception as e:
                st.error(f"模型检测失败: {str(e)}") 