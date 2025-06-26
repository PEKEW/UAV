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
    
    print(f"总共生成 {len(all_segments)} 个数据段")
    return all_segments

def inject_anomalies_to_segments(segments, anomaly_ratio=1/3):
    """
    当注入的异常段时长≥10秒(33%)时，整个30秒段标记为异常
    """
    total_segments = len(segments)
    target_anomaly_count = int(total_segments * anomaly_ratio)
    
    # 随机选择要注入异常的段
    anomaly_indices = random.sample(range(total_segments), target_anomaly_count)
    
    for i, segment in enumerate(segments):
        df = segment['data'].copy()
        segment['original_data'] = df.copy()
        
        if i in anomaly_indices:
            # 注入异常段
            anomaly_duration, modified_df = inject_single_anomaly(df, segment['duration'])
            segment['data'] = modified_df
            segment['anomaly_duration'] = anomaly_duration
            
            if anomaly_duration >= 10:
                segment['label'] = 1
                segment['is_anomaly'] = True
            else:
                segment['label'] = 0
                segment['is_anomaly'] = False
        else:
            segment['label'] = 0
            segment['is_anomaly'] = False
            segment['anomaly_duration'] = 0
    
    actual_anomaly_count = sum(1 for seg in segments if seg['label'] == 1)
    
    if actual_anomaly_count < target_anomaly_count * 0.8:
        # 大概率异常比例过低，需要增加异常注入 
        print(f"实际异常比例过低，尝试增加异常注入")
        normal_indices = [i for i, seg in enumerate(segments) if seg['label'] == 0]
        additional_needed = target_anomaly_count - actual_anomaly_count
        
        if len(normal_indices) >= additional_needed:
            additional_indices = random.sample(normal_indices, additional_needed)
            
            for i in additional_indices:
                segment = segments[i]
                df = segment['original_data'].copy()
                anomaly_duration, modified_df = inject_single_anomaly(df, segment['duration'])
                segment['data'] = modified_df
                segment['anomaly_duration'] = anomaly_duration
                
                if anomaly_duration >= 10:
                    segment['label'] = 1
                    segment['is_anomaly'] = True
                else:
                    segment['label'] = 0
                    segment['is_anomaly'] = False
    
    actual_anomaly_count = sum(1 for seg in segments if seg['label'] == 1)
    print(f"目标异常段数: {target_anomaly_count}")
    print(f"实际异常段数: {actual_anomaly_count} ({actual_anomaly_count/total_segments*100:.1f}%)")
    
    return segments

def inject_single_anomaly(df, segment_duration=30):
    """
    向单个数据段注入异常
    """
    if len(df) == 0:
        return 0, df
    
    anomaly_duration = random.uniform(5, 25)
    
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
        df_modified.loc[anomaly_mask, 'Ecell_V'] -= random.uniform(0.2, 0.5)
        
    elif anomaly_type == 'current_spike' and 'I_mA' in df.columns:
        spike_magnitude = random.uniform(300, 800)
        df_modified.loc[anomaly_mask, 'I_mA'] += spike_magnitude
        
    elif anomaly_type == 'temperature_rise' and 'Temperature__C' in df.columns:
        temp_increase = random.uniform(10, 20)
        df_modified.loc[anomaly_mask, 'Temperature__C'] += temp_increase
        
    elif anomaly_type == 'capacity_fade':
        if 'QCharge_mA_h' in df.columns:
            fade_amount = random.uniform(100, 300)
            df_modified.loc[anomaly_mask, 'QCharge_mA_h'] -= fade_amount
        if 'QDischarge_mA_h' in df.columns:
            df_modified.loc[anomaly_mask, 'QDischarge_mA_h'] += fade_amount * 0.3
            
    elif anomaly_type == 'voltage_fluctuation' and 'Ecell_V' in df.columns:
        fluctuation = np.random.normal(0, 0.1, affected_points)
        df_modified.loc[anomaly_mask, 'Ecell_V'] += fluctuation
    
    return anomaly_duration, df_modified

def create_final_dataset(segments):
    all_data = []
    
    feature_columns = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
                    'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
    
    for i, segment in enumerate(segments):
        df = segment['data']
        
        df = df.copy()
        df['segment_id'] = i
        df['file_id'] = segment['file_id']
        df['label'] = segment['label']
        df['is_anomaly'] = segment['is_anomaly']
        df['anomaly_duration'] = segment['anomaly_duration']
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    final_df = final_df.fillna(0)
    
    total_points = len(final_df)
    anomaly_points = len(final_df[final_df['label'] == 1])
    normal_points = len(final_df[final_df['label'] == 0])
    
    print(f"\n数据集统计:")
    print(f"总数据点: {total_points}")
    print(f"正常数据点: {normal_points} ({normal_points/total_points*100:.1f}%)")
    print(f"异常数据点: {anomaly_points} ({anomaly_points/total_points*100:.1f}%)")
    
    return final_df

def save_dataset_and_segments(final_df, segments, output_dir):
    output_dir = Path(output_dir)
    
    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    dataset_path = output_dir / 'evtol_anomaly_dataset.csv'
    final_df.to_csv(dataset_path, index=False)
    print(f"数据集已保存到: {dataset_path}")
    
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
    
    selected_files = select_random_evtol_files(evtol_data_path, n_files=3)
    print(f"选中的文件: {[f.name for f in selected_files]}")
    
    raw_data_dir = Path(output_dir) / "raw_data"
    raw_data_dir.mkdir(exist_ok=True)
    for file_path in selected_files:
        import shutil
        shutil.copy2(file_path, raw_data_dir / file_path.name)
    
    data_dict = load_evtol_data(selected_files)
    
    segments = slice_data_into_30s_segments(data_dict)
    
    print("\n注入异常")
    segments_with_anomalies = inject_anomalies_to_segments(segments, anomaly_ratio=1/3)
    
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