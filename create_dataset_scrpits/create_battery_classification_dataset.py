#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def create_battery_anomaly_dataset(csv_path, n_points=10000, anomaly_ratio=0.33):
    """从VAH01.csv创建电池异常检测数据集"""
    
    # 读取原始数据
    df_original = pd.read_csv(csv_path)
    print(f"原始数据形状: {df_original.shape}")
    
    # 如果数据点不够，重复采样
    if len(df_original) < n_points:
        repeat_times = (n_points // len(df_original)) + 1
        df_extended = pd.concat([df_original] * repeat_times, ignore_index=True)
        df_selected = df_extended.head(n_points).copy()
    else:
        df_selected = df_original.head(n_points).copy()
    
    # 重置时间序列
    df_selected['time_s'] = np.arange(len(df_selected))
    
    # 初始化标签（0为正常，1为异常）
    df_selected['label'] = 0
    
    # 计算需要的异常数据点数量
    n_anomalies = int(n_points * anomaly_ratio)
    
    # 生成异常数据段
    anomaly_segments = []
    remaining_anomalies = n_anomalies
    
    while remaining_anomalies > 0:
        # 随机选择异常段的起始位置
        start_idx = random.randint(0, n_points - 20)
        # 异常段长度（10-100个点）
        segment_length = min(random.randint(10, 100), remaining_anomalies)
        end_idx = min(start_idx + segment_length, n_points)
        
        # 检查是否与已有异常段重叠
        overlap = False
        for existing_start, existing_end in anomaly_segments:
            if not (end_idx <= existing_start or start_idx >= existing_end):
                overlap = True
                break
        
        if not overlap:
            anomaly_segments.append((start_idx, end_idx))
            remaining_anomalies -= (end_idx - start_idx)
    
    # 应用异常模式到电池数据
    for start_idx, end_idx in anomaly_segments:
        segment_length = end_idx - start_idx
        t_segment = np.linspace(0, 1, segment_length)
        
        # 标记为异常
        df_selected.loc[start_idx:end_idx-1, 'label'] = 1
        
        # 随机选择异常类型
        anomaly_type = random.choice(['voltage_drop', 'current_spike', 'temperature_rise', 'capacity_fade'])
        
        if anomaly_type == 'voltage_drop':
            # 电压骤降
            voltage_drop = 0.3 * t_segment + np.random.normal(0, 0.05, segment_length)
            df_selected.loc[start_idx:end_idx-1, 'Ecell_V'] -= voltage_drop
            
        elif anomaly_type == 'current_spike':
            # 电流异常峰值
            current_spike = 500 * np.sin(t_segment * 4 * np.pi) + np.random.normal(0, 50, segment_length)
            df_selected.loc[start_idx:end_idx-1, 'I_mA'] += current_spike
            
        elif anomaly_type == 'temperature_rise':
            # 温度异常升高
            temp_rise = 15 * t_segment + 5 * np.sin(t_segment * 6 * np.pi) + np.random.normal(0, 2, segment_length)
            df_selected.loc[start_idx:end_idx-1, 'Temperature__C'] += temp_rise
            
        elif anomaly_type == 'capacity_fade':
            # 容量衰减异常
            capacity_fade = 200 * t_segment + np.random.normal(0, 20, segment_length)
            df_selected.loc[start_idx:end_idx-1, 'QCharge_mA_h'] -= capacity_fade
            df_selected.loc[start_idx:end_idx-1, 'QDischarge_mA_h'] += capacity_fade * 0.5
    
    # 添加轻微噪声以增加真实性
    noise_columns = ['Ecell_V', 'I_mA', 'Temperature__C']
    for col in noise_columns:
        if col in df_selected.columns:
            noise_std = df_selected[col].std() * 0.01  # 1%的标准差作为噪声
            df_selected[col] += np.random.normal(0, noise_std, len(df_selected))
    
    return df_selected

def visualize_battery_anomaly_data(df):
    """可视化电池异常检测数据"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 分离正常和异常数据
    normal_data = df[df['label'] == 0]
    anomaly_data = df[df['label'] == 1]
    
    # 1. 电池电压
    axes[0, 0].plot(normal_data['time_s'], normal_data['Ecell_V'], 'b-', alpha=0.7, linewidth=0.8, label='Normal')
    axes[0, 0].scatter(anomaly_data['time_s'], anomaly_data['Ecell_V'], c='red', s=1, alpha=0.8, label='Anomaly')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cell Voltage (V)')
    axes[0, 0].set_title('Battery Cell Voltage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 电流
    axes[0, 1].plot(normal_data['time_s'], normal_data['I_mA'], 'b-', alpha=0.7, linewidth=0.8, label='Normal')
    axes[0, 1].scatter(anomaly_data['time_s'], anomaly_data['I_mA'], c='red', s=1, alpha=0.8, label='Anomaly')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Current (mA)')
    axes[0, 1].set_title('Battery Current')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 温度
    axes[1, 0].plot(normal_data['time_s'], normal_data['Temperature__C'], 'b-', alpha=0.7, linewidth=0.8, label='Normal')
    axes[1, 0].scatter(anomaly_data['time_s'], anomaly_data['Temperature__C'], c='red', s=1, alpha=0.8, label='Anomaly')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Battery Temperature')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 充电容量
    axes[1, 1].plot(normal_data['time_s'], normal_data['QCharge_mA_h'], 'b-', alpha=0.7, linewidth=0.8, label='Normal')
    axes[1, 1].scatter(anomaly_data['time_s'], anomaly_data['QCharge_mA_h'], c='red', s=1, alpha=0.8, label='Anomaly')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Charge Capacity (mAh)')
    axes[1, 1].set_title('Battery Charge Capacity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('battery_anomaly_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计信息
    total_points = len(df)
    anomaly_points = len(df[df['label'] == 1])
    normal_points = len(df[df['label'] == 0])
    
    print(f"\n电池数据统计:")
    print(f"总数据点: {total_points}")
    print(f"正常数据点: {normal_points} ({normal_points/total_points*100:.1f}%)")
    print(f"异常数据点: {anomaly_points} ({anomaly_points/total_points*100:.1f}%)")
    
    # 异常数据特征统计
    print(f"\n异常数据特征:")
    if len(anomaly_data) > 0:
        print(f"电压范围: {anomaly_data['Ecell_V'].min():.3f}V - {anomaly_data['Ecell_V'].max():.3f}V")
        print(f"电流范围: {anomaly_data['I_mA'].min():.1f}mA - {anomaly_data['I_mA'].max():.1f}mA")
        print(f"温度范围: {anomaly_data['Temperature__C'].min():.1f}°C - {anomaly_data['Temperature__C'].max():.1f}°C")

def prepare_data_for_lstm(df, sequence_length=50):
    """为LSTM准备序列数据"""
    
    # 选择特征列
    feature_columns = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h', 
                      'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
    
    # 数据标准化
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # 创建序列数据
    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled[feature_columns].iloc[i-sequence_length:i].values)
        y.append(df_scaled['label'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"LSTM输入数据形状: {X.shape}")
    print(f"LSTM标签数据形状: {y.shape}")
    
    return X, y, scaler

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    print("创建电池异常检测数据集...")
    
    # 创建数据集
    battery_data = create_battery_anomaly_dataset('VAH01.csv', n_points=10000, anomaly_ratio=0.33)
    
    # 保存为CSV
    output_file = 'battery_anomaly_dataset.csv'
    battery_data.to_csv(output_file, index=False)
    print(f"数据集已保存为: {output_file}")
    
    # 可视化
    print("生成可视化图表...")
    visualize_battery_anomaly_data(battery_data)
    
    # 准备LSTM数据
    print("准备LSTM训练数据...")
    X, y, scaler = prepare_data_for_lstm(battery_data)
    
    # 保存预处理后的数据
    np.save('battery_X.npy', X)
    np.save('battery_y.npy', y)
    print("LSTM训练数据已保存为: battery_X.npy, battery_y.npy")