import streamlit as st
import numpy as np

def render_flight_data_controls(chart_area_instance, data):
    """渲染飞行数据控制部分"""
    st.markdown("### 飞行数据选择与分析")
    range_col1, range_col2, range_col3, range_col4 = st.columns(4)
    
    time_min, time_max = data['time'].min(), data['time'].max()
    
    with range_col1:
        start_time = st.number_input(
            "起始时间 (秒)", 
            min_value=float(time_min), 
            max_value=float(time_max), 
            value=float(time_min),
            step=1.0,
            format="%.1f",
            key="flight_start_time"
        )
    
    with range_col2:
        # 使用可视化配置的最大时长作为默认结束时间
        max_viz_duration = chart_area_instance.get_session_state('viz_max_duration', 300)
        default_end = min(float(time_min) + max_viz_duration, float(time_max))
        end_time = st.number_input(
            "结束时间 (秒)", 
            min_value=float(time_min), 
            max_value=float(time_max), 
            value=default_end,
            step=1.0,
            format="%.1f",
            key="flight_end_time"
        )
    
    with range_col3:
        if st.button("应用选择", key="flight_apply_range"):
            if start_time < end_time:
                chart_area_instance.set_session_state('selected_range', (start_time, end_time))
                st.success(f"已选择范围: {start_time:.1f}s - {end_time:.1f}s (时长: {end_time-start_time:.1f}s)")
                st.rerun()
            else:
                st.error("时间范围非法：起始时间必须小于结束时间！")
    
    with range_col4:
        if st.button("重置范围", key="flight_reset_range"):
            chart_area_instance.set_session_state('selected_range', None)
            st.success("已重置为显示全部数据")
            st.rerun()

