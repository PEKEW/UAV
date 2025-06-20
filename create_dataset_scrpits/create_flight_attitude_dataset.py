#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.preprocessing import MinMaxScaler

def generate_flight_attitude_dataset(n_points=10000, anomaly_ratio=0.33):
    """生成飞行姿态异常检测数据集，正常:异常约为2:1"""
    
    # 时间序列
    time = np.arange(n_points)
    
    # 基础飞行轨迹 - 螺旋上升模式
    t_norm = time / n_points * 10 * np.pi  # 归一化时间
    
    # 正常飞行轨迹
    x = 50 * np.cos(t_norm) + 10 * np.sin(t_norm * 0.3)
    y = 50 * np.sin(t_norm) + 10 * np.cos(t_norm * 0.2)
    z = time * 0.02 + 20 * np.sin(t_norm * 0.1)
    
    # 飞行姿态角度（欧拉角）
    roll = 5 * np.sin(t_norm * 0.5) + 2 * np.sin(t_norm * 1.2)
    pitch = 3 * np.cos(t_norm * 0.3) + 1.5 * np.sin(t_norm * 0.8)
    yaw = t_norm * 0.1 + 10 * np.sin(t_norm * 0.2)
    
    # 速度和加速度
    velocity = 25 + 5 * np.sin(t_norm * 0.4)
    acceleration = np.gradient(velocity)
    
    # 高度
    altitude = 100 + z
    
    # 初始化标签（0为正常，1为异常）
    labels = np.zeros(n_points)
    
    # 生成异常数据（约1/3的数据点，使得正常:异常约为2:1）
    n_anomalies = int(n_points * anomaly_ratio)
    anomaly_segments = []
    
    # 创建连续的异常段
    remaining_anomalies = n_anomalies
    while remaining_anomalies > 0:
        # 随机选择异常段的起始位置
        start_idx = random.randint(0, n_points - 20)
        # 异常段长度（10-150个点）
        segment_length = min(random.randint(10, 150), remaining_anomalies)
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
    
    # 应用异常模式到飞行姿态数据
    for start_idx, end_idx in anomaly_segments:
        segment_length = end_idx - start_idx
        t_segment = np.linspace(0, 1, segment_length)
        
        # 标记为异常
        labels[start_idx:end_idx] = 1
        
        # 随机选择异常类型
        anomaly_type = random.choice(['spiral_dive', 'sudden_turn', 'altitude_drop', 'roll_instability', 'stall', 'wind_shear'])
        
        if anomaly_type == 'spiral_dive':
            # 螺旋俯冲
            x[start_idx:end_idx] += 20 * np.cos(t_segment * 10 * np.pi)
            y[start_idx:end_idx] += 20 * np.sin(t_segment * 10 * np.pi)
            z[start_idx:end_idx] -= 30 * t_segment
            roll[start_idx:end_idx] += 45 * np.sin(t_segment * 5 * np.pi)
            pitch[start_idx:end_idx] -= 20 * t_segment
            
        elif anomaly_type == 'sudden_turn':
            # 急转弯
            x[start_idx:end_idx] += 30 * np.sin(t_segment * 3 * np.pi)
            y[start_idx:end_idx] += 30 * np.cos(t_segment * 3 * np.pi)
            yaw[start_idx:end_idx] += 90 * t_segment
            roll[start_idx:end_idx] += 30 * np.sin(t_segment * 4 * np.pi)
            
        elif anomaly_type == 'altitude_drop':
            # 高度骤降
            z[start_idx:end_idx] -= 50 * t_segment
            pitch[start_idx:end_idx] -= 30 * t_segment
            velocity[start_idx:end_idx] += 15 * t_segment
            
        elif anomaly_type == 'roll_instability':
            # 滚转不稳定
            roll[start_idx:end_idx] += 60 * np.sin(t_segment * 8 * np.pi)
            x[start_idx:end_idx] += 15 * np.sin(t_segment * 6 * np.pi)
            y[start_idx:end_idx] += 15 * np.cos(t_segment * 6 * np.pi)
            
        elif anomaly_type == 'stall':
            # 失速
            velocity[start_idx:end_idx] -= 20 * t_segment
            pitch[start_idx:end_idx] += 25 * t_segment
            z[start_idx:end_idx] -= 15 * t_segment
            roll[start_idx:end_idx] += 10 * np.sin(t_segment * 3 * np.pi)
            
        elif anomaly_type == 'wind_shear':
            # 风切变
            x[start_idx:end_idx] += 25 * np.sin(t_segment * 2 * np.pi)
            y[start_idx:end_idx] += 25 * np.cos(t_segment * 2 * np.pi)
            velocity[start_idx:end_idx] += 10 * np.sin(t_segment * 4 * np.pi)
            roll[start_idx:end_idx] += 20 * np.sin(t_segment * 6 * np.pi)
    
    # 重新计算加速度（基于修改后的速度）
    acceleration = np.gradient(velocity)
    
    # 添加噪声
    noise_level = 0.5
    x += np.random.normal(0, noise_level, n_points)
    y += np.random.normal(0, noise_level, n_points)
    z += np.random.normal(0, noise_level * 0.3, n_points)
    roll += np.random.normal(0, 0.2, n_points)
    pitch += np.random.normal(0, 0.2, n_points)
    yaw += np.random.normal(0, 0.1, n_points)
    velocity += np.random.normal(0, 0.5, n_points)
    acceleration += np.random.normal(0, 0.1, n_points)
    altitude = 100 + z
    
    # 创建DataFrame
    df = pd.DataFrame({
        'time': time,
        'x': x,
        'y': y,
        'z': z,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'velocity': velocity,
        'acceleration': acceleration,
        'altitude': altitude,
        'label': labels.astype(int)
    })
    
    return df

def visualize_flight_attitude_data(df):
    """可视化飞行姿态异常检测数据"""
    fig = plt.figure(figsize=(16, 12))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(231, projection='3d')
    
    # 分离正常和异常数据
    normal_data = df[df['label'] == 0]
    anomaly_data = df[df['label'] == 1]
    
    # 绘制轨迹
    ax1.plot(normal_data['x'], normal_data['y'], normal_data['z'], 
             'b-', alpha=0.6, linewidth=1, label='正常飞行')
    ax1.scatter(anomaly_data['x'], anomaly_data['y'], anomaly_data['z'], 
                c='red', s=1, alpha=0.8, label='异常飞行')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D飞行轨迹')
    ax1.legend()
    
    # 姿态角时间序列
    ax2 = fig.add_subplot(232)
    ax2.plot(df['time'], df['roll'], 'g-', alpha=0.7, label='Roll', linewidth=0.8)
    ax2.plot(df['time'], df['pitch'], 'orange', alpha=0.7, label='Pitch', linewidth=0.8)
    ax2.plot(df['time'], df['yaw'], 'purple', alpha=0.7, label='Yaw', linewidth=0.8)
    
    # 标记异常区域
    anomaly_mask = df['label'] == 1
    ax2.fill_between(df['time'], -80, 80, where=anomaly_mask, 
                     color='red', alpha=0.2, label='异常区域')
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('角度 (度)')
    ax2.set_title('飞行姿态角')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 高度时间序列
    ax3 = fig.add_subplot(233)
    ax3.plot(df['time'], df['altitude'], 'b-', alpha=0.7, linewidth=0.8)
    ax3.fill_between(df['time'], 0, 300, where=anomaly_mask, 
                     color='red', alpha=0.2, label='异常区域')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('高度 (m)')
    ax3.set_title('飞行高度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 速度时间序列
    ax4 = fig.add_subplot(234)
    ax4.plot(df['time'], df['velocity'], 'orange', alpha=0.7, linewidth=0.8)
    ax4.fill_between(df['time'], 0, 50, where=anomaly_mask, 
                     color='red', alpha=0.2, label='异常区域')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('速度 (m/s)')
    ax4.set_title('飞行速度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 加速度时间序列
    ax5 = fig.add_subplot(235)
    ax5.plot(df['time'], df['acceleration'], 'purple', alpha=0.7, linewidth=0.8)
    ax5.fill_between(df['time'], -5, 5, where=anomaly_mask, 
                     color='red', alpha=0.2, label='异常区域')
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('加速度 (m/s²)')
    ax5.set_title('飞行加速度')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 异常类型分布（模拟）
    ax6 = fig.add_subplot(236)
    anomaly_types = ['螺旋俯冲', '急转弯', '高度骤降', '滚转不稳定', '失速', '风切变']
    anomaly_counts = [len(anomaly_data) // 6] * 6  # 假设平均分布
    ax6.bar(anomaly_types, anomaly_counts, color='red', alpha=0.7)
    ax6.set_title('异常类型分布')
    ax6.set_ylabel('数量')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('flight_attitude_anomaly_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计信息
    total_points = len(df)
    anomaly_points = len(df[df['label'] == 1])
    normal_points = len(df[df['label'] == 0])
    
    print(f"飞行姿态数据统计:")
    print(f"总数据点: {total_points}")
    print(f"正常数据点: {normal_points} ({normal_points/total_points*100:.1f}%)")
    print(f"异常数据点: {anomaly_points} ({anomaly_points/total_points*100:.1f}%)")
    print(f"正常:异常比例 ≈ {normal_points/anomaly_points:.1f}:1")
    
    # 异常数据特征统计
    print(f"\n异常飞行数据特征:")
    if len(anomaly_data) > 0:
        print(f"Roll角范围: {anomaly_data['roll'].min():.1f}° - {anomaly_data['roll'].max():.1f}°")
        print(f"Pitch角范围: {anomaly_data['pitch'].min():.1f}° - {anomaly_data['pitch'].max():.1f}°")
        print(f"Yaw角范围: {anomaly_data['yaw'].min():.1f}° - {anomaly_data['yaw'].max():.1f}°")
        print(f"速度范围: {anomaly_data['velocity'].min():.1f}m/s - {anomaly_data['velocity'].max():.1f}m/s")
        print(f"高度范围: {anomaly_data['altitude'].min():.1f}m - {anomaly_data['altitude'].max():.1f}m")

def prepare_flight_data_for_lstm(df, sequence_length=50):
    """为LSTM准备飞行姿态序列数据"""
    
    # 选择特征列
    feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
    
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
    
    print(f"飞行姿态LSTM输入数据形状: {X.shape}")
    print(f"飞行姿态LSTM标签数据形状: {y.shape}")
    
    return X, y, scaler

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    print("生成飞行姿态异常检测数据集...")
    
    # 生成数据集
    flight_data = generate_flight_attitude_dataset(n_points=10000, anomaly_ratio=0.33)
    
    # 保存为CSV
    output_file = 'flight_attitude_anomaly_dataset.csv'
    flight_data.to_csv(output_file, index=False)
    print(f"飞行姿态数据集已保存为: {output_file}")
    
    # 可视化
    print("生成可视化图表...")
    visualize_flight_attitude_data(flight_data)
    
    # 准备LSTM数据
    print("准备LSTM训练数据...")
    X, y, scaler = prepare_flight_data_for_lstm(flight_data)
    
    # 保存预处理后的数据
    np.save('flight_X.npy', X)
    np.save('flight_y.npy', y)
    print("飞行姿态LSTM训练数据已保存为: flight_X.npy, flight_y.npy")