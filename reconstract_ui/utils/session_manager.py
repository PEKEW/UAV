"""
Session state management utilities
"""
import streamlit as st

class SessionManager:
    """Manages Streamlit session state initialization"""
    
    @staticmethod
    def init_session_state():
        """Initialize all session state variables"""
        # Current status
        if 'current_status' not in st.session_state:
            st.session_state.current_status = 'offline'
        
        # Data state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'data_type' not in st.session_state:
            st.session_state.data_type = 'unknown'
        if 'selected_time_range' not in st.session_state:
            st.session_state.selected_time_range = None
        if 'show_data_labels' not in st.session_state:
            st.session_state.show_data_labels = False
        
        # Model state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        if 'model_detection_completed' not in st.session_state:
            st.session_state.model_detection_completed = False