import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def select_random_evtol_files(evtol_path, n_files=3):
    evtol_path = Path(evtol_path)
    csv_files = list(evtol_path.glob('*.csv'))
    
    if len(csv_files) < n_files:
        raise ValueError(f"可用文件数({len(csv_files)})少于需要的文件数({n_files})")
    
    selected_files = random.sample(csv_files, n_files)
    return selected_files

def load_evtol_data(file_paths):
    all_data = {}
    
    for file_path in file_paths:
        file_id = file_path.stem
        df = pd.read_csv(file_path)
        
        # 基本数据清理
        df = df.dropna()
        if 'time_s' in df.columns:
            df = df.sort_values('time_s')
        
        all_data[file_id] = df
        print(f"加载文件 {file_id}: {len(df)} 行数据")
    
    return all_data

def slice_data_into_30s_segments(data_dict, time_column='time_s'):
    """将数据按30秒时间窗口切片，滑动步长30秒（无重叠）"""
    all_segments = []
    
    for file_id, df in data_dict.items():
        if time_column not in df.columns:
            continue
            
        df = df.sort_values(time_column).reset_index(drop=True)
        time_data = df[time_column].values
        
        window_size = 30
        start_time = time_data[0]
        end_time = time_data[-1]
        
        current_time = start_time
        segment_id = 0
        
        # test_max_count = 1000
        # test_count = 0
        
        while current_time + window_size <= end_time:
            mask = (time_data >= current_time) & (time_data < current_time + window_size)
            segment_data = df[mask].copy()
            
            if len(segment_data) > 0:
                segment_info = {
                    'file_id': file_id,
                    'segment_id': segment_id,
                    'start_time': current_time,
                    'end_time': current_time + window_size,
                    'data': segment_data,
                    'duration': window_size
                }
                all_segments.append(segment_info)
                segment_id += 1
            current_time += window_size
            # test_count += 1
            # if test_count >= test_max_count:
            #     break
    print(f"总共生成 {len(all_segments)} 个数据段")
    return all_segments

def inject_anomalies_to_segments(segments, anomaly_ratio=1/3):
    """
    直接随机选择样本注入异常，确保异常时长≥10秒
    """
    total_segments = len(segments)
    target_anomaly_count = int(total_segments * anomaly_ratio)
    
    # 随机选择要注入异常的段
    anomaly_indices = random.sample(range(total_segments), target_anomaly_count)
    
    for i, segment in enumerate(segments):
        df = segment['data'].copy()
        segment['original_data'] = df.copy()
        
        if i in anomaly_indices:
            # 直接注入异常，确保异常时长≥10秒
            anomaly_duration, modified_df = inject_single_anomaly_guaranteed(df, segment['duration'])
            segment['data'] = modified_df
            segment['anomaly_duration'] = anomaly_duration
            segment['label'] = 1
            segment['is_anomaly'] = True
        else:
            segment['label'] = 0
            segment['is_anomaly'] = False
            segment['anomaly_duration'] = 0
    
    actual_anomaly_count = sum(1 for seg in segments if seg['label'] == 1)
    print(f"目标异常段数: {target_anomaly_count}")
    print(f"实际异常段数: {actual_anomaly_count} ({actual_anomaly_count/total_segments*100:.1f}%)")
    
    return segments

def inject_single_anomaly_guaranteed(df, segment_duration=30):
    """
    向单个数据段注入异常，确保异常时长≥15秒
    """
    if len(df) == 0:
        return 0, df
    
    # 确保异常时长≥15秒
    anomaly_duration = random.uniform(15, 30)
    
    anomaly_types = ['voltage_drop', 'current_spike', 'temperature_rise', 'capacity_fade', 'voltage_fluctuation']
    anomaly_type = random.choice(anomaly_types)
    
    max_start_time = max(0, segment_duration - anomaly_duration)
    anomaly_start_time = random.uniform(0, max_start_time)
    anomaly_end_time = anomaly_start_time + anomaly_duration
    
    if 'time_s' in df.columns:
        segment_start_time = df['time_s'].min()
        absolute_anomaly_start = segment_start_time + anomaly_start_time
        absolute_anomaly_end = segment_start_time + anomaly_end_time
        
        anomaly_mask = (df['time_s'] >= absolute_anomaly_start) & (df['time_s'] <= absolute_anomaly_end)
    else:
        start_idx = int(len(df) * (anomaly_start_time / segment_duration))
        end_idx = int(len(df) * (anomaly_end_time / segment_duration))
        anomaly_mask = np.zeros(len(df), dtype=bool)
        anomaly_mask[start_idx:end_idx] = True
    
    affected_points = anomaly_mask.sum()
    if affected_points == 0:
        return 0, df
    
    df_modified = df.copy()
    
    if anomaly_type == 'voltage_drop' and 'Ecell_V' in df.columns:
        df_modified.loc[anomaly_mask, 'Ecell_V'] -= random.uniform(0.5, 1.2)
        
    elif anomaly_type == 'current_spike' and 'I_mA' in df.columns:
        spike_magnitude = random.uniform(800, 1500)
        df_modified.loc[anomaly_mask, 'I_mA'] += spike_magnitude
        
    elif anomaly_type == 'temperature_rise' and 'Temperature__C' in df.columns:
        temp_increase = random.uniform(20, 35)
        df_modified.loc[anomaly_mask, 'Temperature__C'] += temp_increase
        
    elif anomaly_type == 'capacity_fade':
        if 'QCharge_mA_h' in df.columns:
            fade_amount = random.uniform(300, 600)
            df_modified.loc[anomaly_mask, 'QCharge_mA_h'] -= fade_amount
        if 'QDischarge_mA_h' in df.columns:
            df_modified.loc[anomaly_mask, 'QDischarge_mA_h'] += fade_amount * 0.5
            
    elif anomaly_type == 'voltage_fluctuation' and 'Ecell_V' in df.columns:
        fluctuation = np.random.normal(0, 0.25, affected_points)
        df_modified.loc[anomaly_mask, 'Ecell_V'] += fluctuation
    
    return anomaly_duration, df_modified

def create_final_dataset(segments):
    """
    创建最终数据集，每个30秒段作为一个样本
    每个样本保持原始时间序列数据，形状为30*9
    """
    all_samples = []
    
    feature_columns = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                    'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
    
    for i, segment in enumerate(segments):
        df = segment['data'].copy()
        
        # 确保每个样本有30个时间点，避免重复行
        if len(df) < 30:
            # 如果数据点不足30个，使用线性插值而不是重复行
            if len(df) >= 2:
                # 使用时间插值创建30个均匀分布的点
                if 'time_s' in df.columns:
                    time_min = df['time_s'].min()
                    time_max = df['time_s'].max()
                    new_times = np.linspace(time_min, time_max, 30)
                    
                    # 对每个特征列进行插值
                    interpolated_data = {'time_s': new_times}
                    for col in feature_columns:
                        if col in df.columns:
                            interpolated_data[col] = np.interp(new_times, df['time_s'], df[col])
                        else:
                            interpolated_data[col] = np.zeros(30)
                    
                    df = pd.DataFrame(interpolated_data)
                else:
                    # 如果没有时间列，使用索引插值
                    indices = np.linspace(0, len(df)-1, 30)
                    interpolated_data = {}
                    for col in feature_columns:
                        if col in df.columns:
                            interpolated_data[col] = np.interp(indices, range(len(df)), df[col])
                        else:
                            interpolated_data[col] = np.zeros(30)
                    df = pd.DataFrame(interpolated_data)
            else:
                # 如果只有1个或0个数据点，使用重复填充而不是跳过
                print(f"警告: 样本 {i} 数据点不足({len(df)}个)，使用重复填充")
                if len(df) == 1:
                    # 如果只有1个数据点，重复30次并添加小的随机噪声
                    single_row = df.iloc[0].copy()
                    repeated_data = []
                    for _ in range(30):
                        new_row = single_row.copy()
                        # 为数值列添加小的随机噪声
                        for col in feature_columns:
                            if col in new_row and pd.notna(new_row[col]):
                                # 添加标准差1%的随机噪声
                                noise = np.random.normal(0, abs(new_row[col]) * 0.01)
                                new_row[col] += noise
                        repeated_data.append(new_row)
                    df = pd.DataFrame(repeated_data)
                else:
                    # 如果0个数据点，创建默认值
                    default_data = {}
                    if 'time_s' in df.columns:
                        default_data['time_s'] = np.linspace(0, 30, 30)
                    for col in feature_columns:
                        default_data[col] = np.zeros(30)
                    df = pd.DataFrame(default_data)
        
        elif len(df) > 30:
            # 如果数据点超过30个，均匀采样30个点（确保不重复）
            indices = np.linspace(0, len(df)-1, 30, dtype=int)
            # 确保索引唯一
            indices = np.unique(indices)
            if len(indices) < 30:
                # 如果去重后不足30个，补充一些索引
                additional_indices = np.random.choice(len(df), 30-len(indices), replace=False)
                indices = np.sort(np.concatenate([indices, additional_indices]))
            else:
                indices = indices[:30]
            df = df.iloc[indices].reset_index(drop=True)
        
        # 确保df正好有30行且无重复
        df = df.head(30).reset_index(drop=True)
        
        # 检查并移除重复行
        if df.duplicated().any():
            print(f"警告: 样本 {i} 存在重复行，正在处理...")
            # 保留第一次出现的行，对重复行添加小的随机噪声
            duplicated_mask = df.duplicated(keep='first')
            for col in feature_columns:
                if col in df.columns and duplicated_mask.any():
                    # 对重复行的数值列添加小的随机噪声
                    noise = np.random.normal(0, df[col].std() * 0.001, duplicated_mask.sum())
                    df.loc[duplicated_mask, col] += noise
        
        # 添加样本元数据
        df['sample_id'] = i
        df['file_id'] = segment['file_id']
        df['label'] = segment['label']
        df['is_anomaly'] = segment['is_anomaly']
        df['anomaly_duration'] = segment['anomaly_duration']
        
        # 确保所有特征列都存在
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # 重新排列列顺序，确保一致性
        base_columns = ['time_s'] if 'time_s' in df.columns else []
        meta_columns = ['sample_id', 'file_id', 'label', 'is_anomaly', 'anomaly_duration']
        feature_columns_exist = [col for col in feature_columns if col in df.columns]
        
        column_order = base_columns + meta_columns + feature_columns_exist
        df = df[column_order]
        
        all_samples.append(df)
    
    # 合并所有样本
    final_df = pd.concat(all_samples, ignore_index=True)
    
    total_samples = len(segments)
    anomaly_samples = sum(1 for seg in segments if seg['label'] == 1)
    normal_samples = sum(1 for seg in segments if seg['label'] == 0)
    
    print(f"\n数据集统计:")
    print(f"总样本数: {total_samples}")
    print(f"正常样本数: {normal_samples} ({normal_samples/total_samples*100:.1f}%)")
    print(f"异常样本数: {anomaly_samples} ({anomaly_samples/total_samples*100:.1f}%)")
    print(f"每个样本形状: 30 * {len(feature_columns_exist)}")
    print(f"数据集总形状: {final_df.shape}")
    
    return final_df

def save_dataset_and_segments(final_df, segments, output_dir):
    output_dir = Path(output_dir)
    
    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # 保存时间序列数据集
    dataset_path = output_dir / 'evtol_anomaly_dataset.csv'
    final_df.to_csv(dataset_path, index=False)
    print(f"时间序列数据集已保存到: {dataset_path}")
    print(f"数据集形状: {final_df.shape}")
    
    # 计算样本统计信息
    unique_samples = final_df['sample_id'].nunique()
    feature_columns = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                    'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
    feature_columns_exist = [col for col in feature_columns if col in final_df.columns]
    print(f"样本数: {unique_samples}")
    print(f"每个样本时间步数: 30")
    print(f"特征数: {len(feature_columns_exist)}")
    print(f"每个样本形状: 30 * {len(feature_columns_exist)}")
    
    # 保存样本级标签信息
    sample_labels = final_df[['sample_id', 'file_id', 'label', 'is_anomaly', 'anomaly_duration']].drop_duplicates()
    sample_labels_path = processed_dir / 'sample_labels.csv'
    sample_labels.to_csv(sample_labels_path, index=False)
    print(f"样本标签信息已保存到: {sample_labels_path}")
    
    # 保存段信息（保持原有格式）
    segment_info = []
    for segment in segments:
        info = {
            'segment_id': segment.get('segment_id', 0),
            'file_id': segment['file_id'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['duration'],
            'label': segment['label'],
            'is_anomaly': segment['is_anomaly'],
            'anomaly_duration': segment['anomaly_duration'],
            'data_points': len(segment['data'])
        }
        segment_info.append(info)
    
    segment_df = pd.DataFrame(segment_info)
    segment_path = processed_dir / 'segment_info.csv'
    segment_df.to_csv(segment_path, index=False)
    print(f"段信息已保存到: {segment_path}")
    
    return dataset_path, segment_path
    
def main():
    random.seed(42)
    np.random.seed(42)
    
    evtol_data_path = "Datasets/EVTOL"
    output_dir = "."
    
    selected_files = select_random_evtol_files(evtol_data_path, n_files=1)
    print(f"选中的文件: {[f.name for f in selected_files]}")
    
    raw_data_dir = Path(output_dir) / "raw_data"
    raw_data_dir.mkdir(exist_ok=True)
    for file_path in selected_files:
        import shutil
        shutil.copy2(file_path, raw_data_dir / file_path.name)
    
    data_dict = load_evtol_data(selected_files)
    
    segments = slice_data_into_30s_segments(data_dict)
    
    print("\n注入异常")
    segments_with_anomalies = inject_anomalies_to_segments(segments, anomaly_ratio=1/2)
    
    final_dataset = create_final_dataset(segments_with_anomalies)
    
    print("\n保存数据...")
    dataset_path, segment_path = save_dataset_and_segments(
        final_dataset, segments_with_anomalies, output_dir
    )
    
    print(f"\nEVTOL异常检测数据集生成完成！")
    print(f"数据集路径: {dataset_path}")
    print(f"段信息路径: {segment_path}")
    
if __name__ == "__main__":
    main()