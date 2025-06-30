"""
Refactored UAV Battery Health Monitoring Application
"""
import streamlit as st
import numpy as np
import torch
import random
import warnings

from .components.header import HeaderComponent
from .components.control_panel import ControlPanelComponent
from .components.visualization import VisualizationComponent
from .handlers.data_handler import DataHandler
from .handlers.model_handler import ModelHandler
from .utils.styles import get_css_styles
from .utils.session_manager import SessionManager

warnings.filterwarnings('ignore')

class BatteryAnalysisApp:
    """Main application class for UAV battery health monitoring"""
    
    def __init__(self):
        self._setup_page_config()
        self._apply_styles()
        self._initialize_components()
        
    def _setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="UAV健康-DEMO", 
            page_icon="",
            layout="wide"
        )
    
    def _apply_styles(self):
        """Apply CSS styles to the application"""
        st.markdown(get_css_styles(), unsafe_allow_html=True)
    
    def _initialize_components(self):
        """Initialize all application components"""
        # Initialize session state
        SessionManager.init_session_state()
        
        # Initialize handlers
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler()
        
        # Initialize UI components
        self.header = HeaderComponent()
        self.control_panel = ControlPanelComponent(self.data_handler, self.model_handler)
        self.visualization = VisualizationComponent()
    
    def run(self):
        """Run the main application"""
        # Render header
        self.header.render()
        
        # Create main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.control_panel.render()
        
        with col2:
            self.visualization.render_chart_area()

def main():
    """Main application entry point"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Run the application
    app = BatteryAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()