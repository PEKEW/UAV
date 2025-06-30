"""
Control panel component for the UAV health monitoring application
"""
import streamlit as st
from ..handlers.data_handler import DataHandler
from ..handlers.model_handler import ModelHandler

class ControlPanelComponent:
    """Handles the control panel UI and interactions"""
    
    def __init__(self, data_handler: DataHandler, model_handler: ModelHandler):
        self.data_handler = data_handler
        self.model_handler = model_handler
    
    def render(self):
        """Render the complete control panel"""
        with st.container():
            st.markdown('<div class="control-panel">控制面板</div>', unsafe_allow_html=True)
            
            self._render_monitoring_mode()
            self._render_data_selection()
            self._render_anomaly_detection()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_monitoring_mode(self):
        """Render monitoring mode selection"""
        st.markdown("### 监测模式 ")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("实时监测", use_container_width=True):
                st.toast("️不行！", icon="⚠️")
                st.session_state.current_status = 'offline'
        
        with col2:
            if st.button("离线监测", use_container_width=True):
                st.session_state.current_status = 'offline'
        
        # Status display
        if st.session_state.current_status == 'online':
            st.markdown('<div class="status-online"> 没有实时监测！</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline">离线监测</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    def _render_data_selection(self):
        """Render data selection section"""
        st.markdown("### 数据选择")
        
        uploaded_file = st.file_uploader(
            "点击导入或拖拽数据文件到此处",
            type=['h5', 'hdf5'],
            help="支持H5格式（battery_开头为电池数据，flight_开头为飞行数据）"
        )
        
        if st.button("导入数据", use_container_width=True, 
                    disabled=not (uploaded_file is not None and st.session_state.current_status == 'offline')):
            self.data_handler.load_data(uploaded_file)
        
        # Data status display
        if st.session_state.data_loaded:
            st.success("数据已导入")
            if hasattr(st.session_state, 'data_type'):
                st.info(f"数据类型: {st.session_state.data_type}")
        else:
            st.info("未导入数据")
        
        st.markdown("---")
    
    def _render_anomaly_detection(self):
        """Render anomaly detection section"""
        st.markdown("### 异常识别")
        st.markdown("#### 模型选择")
        
        uploaded_model = st.file_uploader(
            "点击导入或拖拽模型文件到此处",
            type=['pkl', 'joblib', 'h5', 'pth', 'pt', 'onnx'],
            help="支持pkl, joblib, h5, pth, pt, onnx格式",
            key="model_uploader"
        )
        
        if st.button("加载模型", use_container_width=True,
                    disabled=not (uploaded_model is not None and st.session_state.current_status == 'offline')):
            self.model_handler.load_model(uploaded_model)
        
        # Model status display
        if st.session_state.model_loaded:
            st.success("模型已加载")
            if st.session_state.model_info:
                with st.expander("模型信息"):
                    for key, value in st.session_state.model_info.items():
                        st.write(f"**{key}**: {value}")
        else:
            st.info("未加载模型")
        
        # Start detection button
        if st.button("开始识别(务必确保模型类型和识别特征匹配)", use_container_width=True,
                    disabled=not (st.session_state.data_loaded and st.session_state.model_loaded)):
            self._start_anomaly_detection()
    
    def _start_anomaly_detection(self):
        """Start anomaly detection process"""
        if st.session_state.model_loaded and st.session_state.data_loaded:
            model_type = st.session_state.model.get('model_type', 'unknown')
            data_type = st.session_state.data_type
            
            is_compatible, compatibility_msg = self.model_handler.validate_model_data_compatibility(model_type, data_type)
            
            if not is_compatible:
                st.error(f"模型与数据不兼容: {compatibility_msg}")
                st.error("请确保:")
                st.error("1. 电池模型文件名以 'battery_' 开头，用于电池数据")
                st.error("2. 飞行模型文件名以 'flight_' 开头，用于飞行数据")
                return
            
            with st.spinner('正在执行模型识别...'):
                if hasattr(st.session_state, 'data'):
                    anomaly_regions = self.model_handler.generate_anomaly_detection(st.session_state.data['time'])
                    st.session_state.data['anomaly_regions'] = anomaly_regions
                    st.session_state.model_detection_completed = True