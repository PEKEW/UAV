import streamlit as st
from .components.control_panel import ControlPanel
from .components.chart_area import ChartArea

class BatteryAnalysisApp:
    """电池健康管理应用主类"""
    
    def __init__(self):
        self.setup_page()
        self.control_panel = ControlPanel()
        self.chart_area = ChartArea()
    
    def setup_page(self):
        st.set_page_config(
            page_title="电池|飞行姿态健康管理-DEMO", 
            page_icon="",
            layout="wide"
        )
        
        self.setup_styles()
    
    def setup_styles(self):
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
    
    def create_header(self):
        st.markdown('<h1 class="main-header">电池|飞行姿态健康管理-DEMO</h1>', unsafe_allow_html=True)
    
    def run(self):
        self.create_header()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.control_panel.render()
        
        with col2:
            self.chart_area.render() 