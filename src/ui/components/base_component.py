import streamlit as st
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """UI组件的基类，定义了基本的组件接口"""
    
    def __init__(self):
        """初始化组件"""
        self.init_session_state()
    
    @abstractmethod
    def init_session_state(self):
        """初始化组件所需的session state"""
        pass
    
    @abstractmethod
    def render(self):
        """渲染组件"""
        pass
    
    def get_session_state(self, key, default=None):
        """安全地获取session state值"""
        return getattr(st.session_state, key, default)
    
    def set_session_state(self, key, value):
        """安全地设置session state值"""
        setattr(st.session_state, key, value) 