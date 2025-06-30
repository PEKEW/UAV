"""
Streamlit CSS styles for the UAV health monitoring application
"""

def get_css_styles():
    """Returns the CSS styles for the application"""
    return """
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
            font-size: 28px !important;
            font-weight: bold !important;
            color: #1f77b4 !important;
            text-align: left;
            margin-bottom: 30px;
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
    </style>
    """