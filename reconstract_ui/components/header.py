"""
Header component for the UAV health monitoring application
"""
import streamlit as st

class HeaderComponent:
    """Handles the application header"""
    
    @staticmethod
    def render():
        """Render the application header"""
        st.markdown('<h1 class="main-header">电池健康管理-DEMO</h1>', unsafe_allow_html=True)