#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import psutil
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def print_memory_usage(stage=""):
    """打印内存使用情况"""
    memory_mb = get_memory_usage()
    print(f"内存使用 {stage}: {memory_mb:.1f} MB")

def generate_flight_attitude_dataset(n_samples, window_size=30, anomaly_ratio=1/3, output_file=None):
    """使用流式处理生成飞行姿态数据集，直接保存到文件"""
    points_per_sample = window_size * 10
    total_points = n_samples * points_per_sample
    
    print(f"生成飞行姿态数据集: {n_samples:,} 个样本, 每个样本 {window_size} 秒")
    print(f"总数据点: {total_points:,}")
    
    # 分块处理，每次处理1000个样本（更小的块）
    chunk_size = 1000
    all_chunks = []
    
    # 如果指定了输出文件，直接写入文件
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        first_chunk = True
    
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_samples = chunk_end - chunk_start
        
        print(f"处理样本 {chunk_start:,} - {chunk_end:,} ({chunk_samples:,} 个样本)")
        
        # 计算当前块的数据点
        chunk_points = chunk_samples * points_per_sample
        chunk_start_point = chunk_start * points_per_sample
        
        # 生成时间序列
        time = np.arange(chunk_start_point, chunk_start_point + chunk_points) / 10
        t_norm = time / (n_samples * points_per_sample / 10) * 10 * np.pi
        
        # 生成基础轨迹
        x = 50 * np.cos(t_norm) + 10 * np.sin(t_norm * 0.3)
        y = 50 * np.sin(t_norm) + 10 * np.cos(t_norm * 0.2)
        z = time * 0.02 + 20 * np.sin(t_norm * 0.1)
        
        # 飞行姿态角度
        roll = 5 * np.sin(t_norm * 0.5) + 2 * np.sin(t_norm * 1.2)
        pitch = 3 * np.cos(t_norm * 0.3) + 1.5 * np.sin(t_norm * 0.8)
        yaw = t_norm * 0.1 + 10 * np.sin(t_norm * 0.2)
        
        # 速度和加速度
        velocity = 25 + 5 * np.sin(t_norm * 0.4)
        acceleration = np.gradient(velocity)
        
        # 高度
        altitude = 100 + z
        
        # 初始化标签
        labels = np.zeros(chunk_points)
        
        # 为当前块注入异常
        n_anomaly_samples = int(chunk_samples * anomaly_ratio)
        anomaly_sample_indices = random.sample(range(chunk_samples), n_anomaly_samples)
        
        for sample_idx in anomaly_sample_indices:
            start_point = sample_idx * points_per_sample
            end_point = (sample_idx + 1) * points_per_sample
            
            total_anomaly_duration = 0
            # 增加目标异常持续时间，从60%提升到80%
            target_anomaly_duration = window_size * 0.8
            
            while total_anomaly_duration < target_anomaly_duration:
                remaining_duration = target_anomaly_duration - total_anomaly_duration
                # 增加异常持续时间范围，从5-15秒改为8-20秒
                anomaly_duration = min(random.uniform(8, 20), remaining_duration)
                
                max_start_offset = window_size - anomaly_duration
                if max_start_offset <= 0:
                    break
                    
                anomaly_start_offset = random.uniform(0, max_start_offset)
                
                anomaly_start_point = start_point + int(anomaly_start_offset * 10)
                anomaly_end_point = anomaly_start_point + int(anomaly_duration * 10)
                anomaly_end_point = min(anomaly_end_point, end_point)
                
                labels[anomaly_start_point:anomaly_end_point] = 1
                total_anomaly_duration += anomaly_duration
            
            sample_anomaly_mask = labels[start_point:end_point] == 1
            anomaly_indices = np.where(sample_anomaly_mask)[0] + start_point
            
            if len(anomaly_indices) > 0:
                anomaly_type = random.choice(['spiral_dive', 'sudden_turn', 'altitude_drop', 
                                             'roll_instability', 'stall', 'wind_shear'])
                
                segment_length = len(anomaly_indices)
                t_segment = np.linspace(0, 1, segment_length)
                
                if anomaly_type == 'spiral_dive':
                    x[anomaly_indices] += 20 * np.cos(t_segment * 10 * np.pi)
                    y[anomaly_indices] += 20 * np.sin(t_segment * 10 * np.pi)
                    z[anomaly_indices] -= 30 * t_segment
                    roll[anomaly_indices] += 45 * np.sin(t_segment * 5 * np.pi)
                    pitch[anomaly_indices] -= 20 * t_segment
                    
                elif anomaly_type == 'sudden_turn':
                    x[anomaly_indices] += 30 * np.sin(t_segment * 3 * np.pi)
                    y[anomaly_indices] += 30 * np.cos(t_segment * 3 * np.pi)
                    yaw[anomaly_indices] += 90 * t_segment
                    roll[anomaly_indices] += 30 * np.sin(t_segment * 4 * np.pi)
                    
                elif anomaly_type == 'altitude_drop':
                    z[anomaly_indices] -= 50 * t_segment
                    pitch[anomaly_indices] -= 30 * t_segment
                    velocity[anomaly_indices] += 15 * t_segment
                    
                elif anomaly_type == 'roll_instability':
                    roll[anomaly_indices] += 60 * np.sin(t_segment * 8 * np.pi)
                    x[anomaly_indices] += 15 * np.sin(t_segment * 6 * np.pi)
                    y[anomaly_indices] += 15 * np.cos(t_segment * 6 * np.pi)
                    
                elif anomaly_type == 'stall':
                    velocity[anomaly_indices] -= 20 * t_segment
                    pitch[anomaly_indices] += 25 * t_segment
                    z[anomaly_indices] -= 15 * t_segment
                    roll[anomaly_indices] += 10 * np.sin(t_segment * 3 * np.pi)
                    
                elif anomaly_type == 'wind_shear':
                    x[anomaly_indices] += 25 * np.sin(t_segment * 2 * np.pi)
                    y[anomaly_indices] += 25 * np.cos(t_segment * 2 * np.pi)
                    velocity[anomaly_indices] += 10 * np.sin(t_segment * 4 * np.pi)
                    roll[anomaly_indices] += 20 * np.sin(t_segment * 6 * np.pi)
        
        # 重新计算加速度和高度
        acceleration = np.gradient(velocity)
        altitude = 100 + z
        
        # 添加噪声
        noise_level = 0.5
        x += np.random.normal(0, noise_level, chunk_points)
        y += np.random.normal(0, noise_level, chunk_points)
        z += np.random.normal(0, noise_level * 0.3, chunk_points)
        roll += np.random.normal(0, 0.2, chunk_points)
        pitch += np.random.normal(0, 0.2, chunk_points)
        yaw += np.random.normal(0, 0.1, chunk_points)
        velocity += np.random.normal(0, 0.5, chunk_points)
        acceleration += np.random.normal(0, 0.1, chunk_points)
        
        # 创建当前块的DataFrame
        chunk_df = pd.DataFrame({
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
        
        # 如果指定了输出文件，直接写入文件
        if output_file:
            if first_chunk:
                chunk_df.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk_df.to_csv(output_file, index=False, mode='a', header=False)
            print(f"已写入文件: {chunk_start:,} - {chunk_end:,} 样本")
        else:
            all_chunks.append(chunk_df)
        
        print_memory_usage(f"块 {chunk_start//chunk_size + 1}")
        
        # 清理内存
        del chunk_df, time, x, y, z, roll, pitch, yaw, velocity, acceleration, altitude, labels
    
    if output_file:
        print(f"数据已直接保存到: {output_file}")
        return output_file
    else:
        # 合并所有块（仅用于小数据集）
        print("合并所有数据块...")
        final_df = pd.concat(all_chunks, ignore_index=True)
        print_memory_usage("合并后")
        return final_df

def create_sample_based_dataset_from_chunk(df_chunk, start_sample_id, window_size=30, sampling_rate=10):
    """从数据块创建样本"""
    points_per_sample = window_size * sampling_rate
    total_points = len(df_chunk)
    n_samples = total_points // points_per_sample
    
    samples = []
    for i in range(n_samples):
        start_idx = i * points_per_sample
        end_idx = (i + 1) * points_per_sample
        
        sample_data = df_chunk.iloc[start_idx:end_idx].copy()
        
        anomaly_points = sample_data['label'].sum()
        anomaly_ratio = anomaly_points / len(sample_data)
        
        # 样本级标签：异常点≥20%则标记为异常样本（降低阈值）
        sample_label = 1 if anomaly_ratio >= 0.2 else 0
        
        sample_info = {
            'sample_id': start_sample_id + i,
            'start_time': sample_data['time'].iloc[0],
            'end_time': sample_data['time'].iloc[-1],
            'window_size': window_size,
            'anomaly_ratio': anomaly_ratio,
            'sample_label': sample_label,
            'data_points': len(sample_data)
        }
        samples.append(sample_info)
    
    return samples

def save_sample_info(samples, output_dir):
    """保存样本信息"""
    output_dir = Path(output_dir)
    
    # 检查异常比例，如果过低则动态调整
    total_samples = len(samples)
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    current_anomaly_ratio = anomaly_samples / total_samples
    
    print(f"当前异常比例: {current_anomaly_ratio:.1%}")
    
    # 如果异常比例低于25%，动态调整阈值
    if current_anomaly_ratio < 0.25:
        print("异常比例过低，动态调整阈值...")
        
        # 按异常比例排序
        sorted_samples = sorted(samples, key=lambda x: x['anomaly_ratio'], reverse=True)
        
        # 计算需要的异常样本数
        target_anomaly_samples = int(total_samples * 0.33)
        
        # 重新标记
        for i, sample in enumerate(sorted_samples):
            if i < target_anomaly_samples:
                sample['sample_label'] = 1
            else:
                sample['sample_label'] = 0
        
        # 重新统计
        anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
        current_anomaly_ratio = anomaly_samples / total_samples
        print(f"调整后异常比例: {current_anomaly_ratio:.1%}")
    
    sample_info = []
    for sample in samples:
        info = {
            'sample_id': sample['sample_id'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'window_size': sample['window_size'],
            'anomaly_ratio': sample['anomaly_ratio'],
            'sample_label': sample['sample_label'],
            'data_points': sample['data_points']
        }
        sample_info.append(info)
    
    sample_df = pd.DataFrame(sample_info)
    sample_path = output_dir / 'sample_info.csv'
    sample_df.to_csv(sample_path, index=False)
    print(f"样本信息已保存到: {sample_path}")
    
    # 统计信息
    normal_samples = total_samples - anomaly_samples
    
    print(f"\n样本统计:")
    print(f"总样本数: {total_samples:,}")
    print(f"正常样本: {normal_samples:,} ({normal_samples/total_samples*100:.1f}%)")
    print(f"异常样本: {anomaly_samples:,} ({anomaly_samples/total_samples*100:.1f}%)")

def create_sample_based_dataset(df, window_size=30, sampling_rate=10):
    """
    每个样本对应一个30秒的时间窗口
    """
    points_per_sample = window_size * sampling_rate
    total_points = len(df)
    n_samples = total_points // points_per_sample
    
    samples = []
    for i in range(n_samples):
        start_idx = i * points_per_sample
        end_idx = (i + 1) * points_per_sample
        
        sample_data = df.iloc[start_idx:end_idx].copy()
        
        anomaly_points = sample_data['label'].sum()
        anomaly_ratio = anomaly_points / len(sample_data)
        
        # 样本级标签：异常点≥20%则标记为异常样本（降低阈值）
        sample_label = 1 if anomaly_ratio >= 0.2 else 0
        
        sample_info = {
            'sample_id': i,
            'start_time': sample_data['time'].iloc[0],
            'end_time': sample_data['time'].iloc[-1],
            'window_size': window_size,
            'anomaly_ratio': anomaly_ratio,
            'sample_label': sample_label,
            'data': sample_data
        }
        samples.append(sample_info)
    
    return samples

def create_final_flight_dataset(samples):
    """使用分块处理创建最终飞行数据集"""
    print("开始创建最终数据集...")
    
    feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
    
    # 分块处理
    chunk_size = 1000  # 每次处理1000个样本
    all_chunks = []
    
    for i in range(0, len(samples), chunk_size):
        chunk_samples = samples[i:i+chunk_size]
        chunk_data = []
        
        for sample in chunk_samples:
            df = sample['data'].copy()
            df['sample_id'] = sample['sample_id']
            df['sample_label'] = sample['sample_label']
            df['anomaly_ratio'] = sample['anomaly_ratio']
            chunk_data.append(df)
        
        # 合并当前块
        if chunk_data:
            chunk_df = pd.concat(chunk_data, ignore_index=True)
            all_chunks.append(chunk_df)
        
        print(f"已处理 {min(i+chunk_size, len(samples))}/{len(samples)} 个样本")
        print_memory_usage(f"块 {i//chunk_size + 1}")
    
    # 最终合并
    print("合并所有数据块...")
    final_df = pd.concat(all_chunks, ignore_index=True)
    
    # 统计信息
    total_samples = len(samples)
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    normal_samples = total_samples - anomaly_samples
    
    total_points = len(final_df)
    anomaly_points = len(final_df[final_df['label'] == 1])
    normal_points = total_points - anomaly_points
    
    print(f"\n数据集统计:")
    print(f"总样本数: {total_samples:,}")
    print(f"正常样本: {normal_samples:,} ({normal_samples/total_samples*100:.1f}%)")
    print(f"异常样本: {anomaly_samples:,} ({anomaly_samples/total_samples*100:.1f}%)")
    print(f"总数据点: {total_points:,}")
    print(f"正常数据点: {normal_points:,} ({normal_points/total_points*100:.1f}%)")
    print(f"异常数据点: {anomaly_points:,} ({anomaly_points/total_points*100:.1f}%)")
    
    return final_df, samples

def save_flight_dataset(final_df, samples, output_dir):
    """使用分块保存飞行数据集"""
    output_dir = Path(output_dir)
    
    # 分块保存数据集
    dataset_path = output_dir / 'flight_anomaly_dataset.csv'
    print(f"开始保存数据集到: {dataset_path}")
    
    # 使用分块写入
    chunk_size = 50000  # 每次写入5万行
    for i in range(0, len(final_df), chunk_size):
        chunk = final_df.iloc[i:i+chunk_size]
        if i == 0:
            chunk.to_csv(dataset_path, index=False, mode='w')
        else:
            chunk.to_csv(dataset_path, index=False, mode='a', header=False)
        
        print(f"已保存 {min(i+chunk_size, len(final_df)):,}/{len(final_df):,} 行数据")
    
    print(f"飞行数据集已保存到: {dataset_path}")
    
    # 保存样本信息
    print("保存样本信息...")
    save_sample_info(samples, output_dir)
    print_memory_usage("保存完成后")

def get_evtol_dataset_size():
    try:
        evtol_segment_path = "../evtol_dataset/processed/segment_info.csv"
        if Path(evtol_segment_path).exists():
            segment_df = pd.read_csv(evtol_segment_path)
            return len(segment_df)
        else:
            current_segment_path = "processed/segment_info.csv"
            if Path(current_segment_path).exists():
                segment_df = pd.read_csv(current_segment_path)
                return len(segment_df)
            else:
                return 100000
    except Exception as e:
        print(f"读取EVTOL数据集信息失败: {e}，使用默认样本数")
        return 100000

def main():
    random.seed(42)
    np.random.seed(42)
    
    print_memory_usage("启动时")
    
    output_dir = "."
    
    print("开始生成数据集")
    
    print("\n获取目标样本数量")
    target_samples = get_evtol_dataset_size()
    print(f"目标样本数: {target_samples:,}")
    print_memory_usage("获取样本数后")
    
    # 直接保存到文件
    output_file = Path(output_dir) / 'flight_anomaly_dataset.csv'
    
    print("\n生成飞行姿态数据")
    generate_flight_attitude_dataset(
        n_samples=target_samples,
        window_size=30,
        anomaly_ratio=1/3,
        output_file=output_file
    )
    print_memory_usage("生成数据后")
    
    print("\n读取数据并创建样本")
    # 分块读取数据
    chunk_size = 50000
    samples = []
    sample_id = 0
    
    for chunk in pd.read_csv(output_file, chunksize=chunk_size):
        chunk_samples = create_sample_based_dataset_from_chunk(chunk, sample_id, window_size=30)
        samples.extend(chunk_samples)
        sample_id += len(chunk_samples)
        print(f"已处理 {len(samples):,} 个样本")
        print_memory_usage(f"读取块 {sample_id//chunk_size + 1}")
    
    print_memory_usage("创建样本后")
    
    print("\n保存样本信息")
    save_sample_info(samples, output_dir)
    print_memory_usage("保存完成后")
    
    print(f"\n数据集生成完成！")
    print(f"总样本数: {len(samples):,}")
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    print(f"异常样本: {anomaly_samples:,} ({anomaly_samples/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    main()