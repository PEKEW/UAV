"""
双域异常检测模型包
包含电池和飞行数据的CNN-LSTM混合模型
"""

from .battery_cnn_lstm import BatteryAnomalyNet
from .flight_cnn_lstm import FlightAnomalyNet
from .base_model import BaseAnomalyModel
from .simple_battery_cnn import SimpleBatteryCNN, SimpleBatteryCNNWithAttention

__all__ = [
    'BatteryAnomalyNet', 
    'FlightAnomalyNet', 
    'BaseAnomalyModel',
    'SimpleBatteryCNN',
    'SimpleBatteryCNNWithAttention'
]