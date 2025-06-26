import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import psutil
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def print_memory_usage(stage=""):
    """打印内存使用情况"""
    memory_mb = get_memory_usage()
    print(f"内存使用 {stage}: {memory_mb:.1f} MB")

def generate_flight_attitude_dataset(n_samples, window_size=30, anomaly_ratio=1/2, output_file=None):
    """使用流式处理生成飞行姿态数据集，直接保存到文件"""
    points_per_sample = window_size * 10
    total_points = n_samples * points_per_sample
    
    print(f"生成飞行姿态数据集: {n_samples:,} 个样本, 每个样本 {window_size} 秒")
    print(f"总数据点: {total_points:,}")
    
    # 分块处理，每次处理1000个样本（更小的块）
    chunk_size = 2000
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
            
            # 直接注入异常，确保异常时长≥10秒
            anomaly_duration = random.uniform(10, 25)  # 确保异常时长≥10秒
            
            max_start_offset = window_size - anomaly_duration
            if max_start_offset <= 0:
                continue
                
            anomaly_start_offset = random.uniform(0, max_start_offset)
            
            anomaly_start_point = start_point + int(anomaly_start_offset * 10)
            anomaly_end_point = anomaly_start_point + int(anomaly_duration * 10)
            anomaly_end_point = min(anomaly_end_point, end_point)
            
            labels[anomaly_start_point:anomaly_end_point] = 1
            
            anomaly_indices = np.arange(anomaly_start_point, anomaly_end_point)
            
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
        noise_level = 0.6
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

def create_sample_based_dataset(df, window_size=30, sampling_rate=10):
    """
    每个样本对应一个30秒的时间窗口，与battery数据集逻辑保持一致
    """
    points_per_sample = window_size * sampling_rate
    total_points = len(df)
    n_samples = total_points // points_per_sample
    
    samples = []
    for i in range(n_samples):
        start_idx = i * points_per_sample
        end_idx = (i + 1) * points_per_sample
        
        sample_data = df.iloc[start_idx:end_idx].copy()
        
        # 计算异常持续时间（秒）
        anomaly_points = sample_data['label'].sum()
        anomaly_duration = anomaly_points / sampling_rate  # 转换为秒
        
        # 与battery数据集一致的异常判断标准：异常时长≥10秒(33%)时标记为异常
        sample_label = 1 if anomaly_duration >= 10 else 0
        
        sample_info = {
            'sample_id': i,
            'start_time': sample_data['time'].iloc[0],
            'end_time': sample_data['time'].iloc[-1],
            'window_size': window_size,
            'anomaly_duration': anomaly_duration,
            'sample_label': sample_label,
            'data_points': len(sample_data),
            'data': sample_data
        }
        samples.append(sample_info)
    
    return samples

def save_sample_info(samples, output_dir):
    """保存样本信息"""
    output_dir = Path(output_dir)
    
    # 统计异常比例
    total_samples = len(samples)
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    current_anomaly_ratio = anomaly_samples / total_samples
    
    print(f"异常比例: {current_anomaly_ratio:.1%}")
    
    sample_info = []
    for sample in samples:
        info = {
            'sample_id': sample['sample_id'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'window_size': sample['window_size'],
            'anomaly_duration': sample['anomaly_duration'],
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

def create_final_flight_dataset(samples):
    """创建最终飞行数据集，与battery数据集格式保持一致"""
    print("开始创建最终数据集...")
    
    feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
    
    all_samples = []
    
    for i, sample in enumerate(samples):
        df = sample['data']
        
        # 确保每个样本有30个时间点
        if len(df) < 30:
            # 如果数据点不足30个，通过插值或重复来补充
            df = df.reset_index(drop=True)
            while len(df) < 30:
                df = pd.concat([df, df.iloc[-1:]], ignore_index=True)
        elif len(df) > 30:
            # 如果数据点超过30个，均匀采样30个点
            indices = np.linspace(0, len(df)-1, 30, dtype=int)
            df = df.iloc[indices].reset_index(drop=True)
        
        # 确保df正好有30行
        df = df.head(30).reset_index(drop=True)
        
        # 添加样本元数据
        df['sample_id'] = sample['sample_id']
        df['sample_label'] = sample['sample_label']
        df['anomaly_duration'] = sample['anomaly_duration']
        
        # 确保所有特征列都存在
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # 重新排列列顺序，确保一致性
        base_columns = ['time'] if 'time' in df.columns else []
        meta_columns = ['sample_id', 'sample_label', 'anomaly_duration']
        feature_columns_exist = [col for col in feature_columns if col in df.columns]
        
        column_order = base_columns + meta_columns + feature_columns_exist
        df = df[column_order]
        
        all_samples.append(df)
    
    # 合并所有样本
    final_df = pd.concat(all_samples, ignore_index=True)
    
    # 统计信息
    total_samples = len(samples)
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    normal_samples = total_samples - anomaly_samples
    
    print(f"\n数据集统计:")
    print(f"总样本数: {total_samples:,}")
    print(f"正常样本: {normal_samples:,} ({normal_samples/total_samples*100:.1f}%)")
    print(f"异常样本: {anomaly_samples:,} ({anomaly_samples/total_samples*100:.1f}%)")
    print(f"每个样本形状: 30 * {len(feature_columns_exist)}")
    print(f"数据集总形状: {final_df.shape}")
    
    return final_df, samples

def save_flight_dataset(final_df, samples, output_dir):
    """保存飞行数据集，与battery数据集格式保持一致"""
    output_dir = Path(output_dir)
    
    # 保存时间序列数据集
    dataset_path = output_dir / 'flight_anomaly_dataset.csv'
    final_df.to_csv(dataset_path, index=False)
    print(f"时间序列数据集已保存到: {dataset_path}")
    print(f"数据集形状: {final_df.shape}")
    
    # 计算样本统计信息
    unique_samples = final_df['sample_id'].nunique()
    feature_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'velocity', 'acceleration', 'altitude']
    feature_columns_exist = [col for col in feature_columns if col in final_df.columns]
    print(f"样本数: {unique_samples}")
    print(f"每个样本时间步数: 30")
    print(f"特征数: {len(feature_columns_exist)}")
    print(f"每个样本形状: 30 * {len(feature_columns_exist)}")
    
    # 保存样本级标签信息
    sample_labels = final_df[['sample_id', 'sample_label', 'anomaly_duration']].drop_duplicates()
    sample_labels_path = output_dir / 'processed'
    sample_labels_path.mkdir(exist_ok=True)
    sample_labels_path = sample_labels_path / 'sample_labels.csv'
    sample_labels.to_csv(sample_labels_path, index=False)
    print(f"样本标签信息已保存到: {sample_labels_path}")
    
    # 保存样本信息
    print("保存样本信息...")
    save_sample_info(samples, output_dir)
    print_memory_usage("保存完成后")

def get_evtol_dataset_size():
    """获取battery数据集的样本数量"""
    try:
        # 尝试读取battery数据集的段信息
        battery_segment_paths = [
            "processed/segment_info.csv"
        ]
        
        for path in battery_segment_paths:
            if Path(path).exists():
                segment_df = pd.read_csv(path)
                sample_count = len(segment_df)
                print(f"从 {path} 读取到 {sample_count} 个样本")
                return sample_count
        
        # 如果找不到battery数据集，使用默认值
        print("未找到battery数据集，使用默认样本数")
        return 20000
    except Exception as e:
        print(f"读取battery数据集信息失败: {e}，使用默认样本数")
        return 20000

def main():
    random.seed(42)
    np.random.seed(42)
    
    print_memory_usage("启动时")
    
    output_dir = "."
    
    print("开始生成飞行数据集")
    
    print("\n获取目标样本数量")
    target_samples = get_evtol_dataset_size()
    print(f"目标样本数: {target_samples:,}")
    print_memory_usage("获取样本数后")
    
    # 生成飞行姿态数据
    print("\n生成飞行姿态数据")
    flight_data = generate_flight_attitude_dataset(
        n_samples=target_samples,
        window_size=30,
        anomaly_ratio=1/2
    )
    print_memory_usage("生成数据后")
    
    print("\n创建样本级数据集")
    samples = create_sample_based_dataset(flight_data, window_size=30)
    print_memory_usage("创建样本后")
    
    print("\n创建最终数据集")
    final_df, samples = create_final_flight_dataset(samples)
    print_memory_usage("创建最终数据集后")
    
    print("\n保存数据集")
    save_flight_dataset(final_df, samples, output_dir)
    
    print(f"\n飞行数据集生成完成！")
    print(f"总样本数: {len(samples):,}")
    anomaly_samples = sum(1 for s in samples if s['sample_label'] == 1)
    print(f"异常样本: {anomaly_samples:,} ({anomaly_samples/len(samples)*100:.1f}%)")
    print(f"每个样本形状: 30 * 9")

if __name__ == "__main__":
    main()