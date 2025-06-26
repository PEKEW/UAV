#!/usr/bin/env python3
"""
重新生成数据集并检查比例
"""
import os
import pandas as pd
from pathlib import Path

def check_evtol_dataset():
    """检查EVTOL数据集"""
    print("="*50)
    print("检查EVTOL数据集")
    print("="*50)
    
    # 检查数据集文件
    dataset_path = "evtol_dataset/evtol_anomaly_dataset.csv"
    segment_path = "evtol_dataset/processed/segment_info.csv"
    
    if Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)
        total_points = len(df)
        anomaly_points = len(df[df['label'] == 1])
        normal_points = total_points - anomaly_points
        
        print(f"EVTOL数据集统计:")
        print(f"  总数据点: {total_points}")
        print(f"  正常数据点: {normal_points} ({normal_points/total_points*100:.1f}%)")
        print(f"  异常数据点: {anomaly_points} ({anomaly_points/total_points*100:.1f}%)")
    else:
        print("EVTOL数据集文件不存在")
        return None
    
    if Path(segment_path).exists():
        segment_df = pd.read_csv(segment_path)
        total_segments = len(segment_df)
        anomaly_segments = len(segment_df[segment_df['label'] == 1])
        
        print(f"EVTOL段统计:")
        print(f"  总段数: {total_segments}")
        print(f"  异常段数: {anomaly_segments} ({anomaly_segments/total_segments*100:.1f}%)")
        
        return total_segments
    else:
        print("EVTOL段信息文件不存在")
        return None

def check_flight_dataset():
    """检查飞行数据集"""
    print("\n" + "="*50)
    print("检查飞行数据集")
    print("="*50)
    
    # 检查数据集文件
    dataset_path = "flight_dataset/flight_anomaly_dataset.csv"
    sample_path = "flight_dataset/sample_info.csv"
    
    if Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)
        total_points = len(df)
        anomaly_points = len(df[df['label'] == 1])
        normal_points = total_points - anomaly_points
        
        print(f"飞行数据集统计:")
        print(f"  总数据点: {total_points}")
        print(f"  正常数据点: {normal_points} ({normal_points/total_points*100:.1f}%)")
        print(f"  异常数据点: {anomaly_points} ({anomaly_points/total_points*100:.1f}%)")
    else:
        print("飞行数据集文件不存在")
        return None
    
    if Path(sample_path).exists():
        sample_df = pd.read_csv(sample_path)
        total_samples = len(sample_df)
        anomaly_samples = len(sample_df[sample_df['sample_label'] == 1])
        
        print(f"飞行样本统计:")
        print(f"  总样本数: {total_samples}")
        print(f"  异常样本数: {anomaly_samples} ({anomaly_samples/total_samples*100:.1f}%)")
        
        return total_samples
    else:
        print("飞行样本信息文件不存在")
        return None

def main():
    """主函数"""
    print("开始重新生成数据集并检查比例...")
    
    # 1. 重新生成EVTOL数据集
    print("\n1. 重新生成EVTOL数据集...")
    os.chdir("evtol_dataset")
    os.system("python create_evtol_dataset.py")
    os.chdir("..")
    
    # 2. 检查EVTOL数据集
    evtol_segments = check_evtol_dataset()
    
    # 3. 重新生成飞行数据集
    print("\n2. 重新生成飞行数据集...")
    os.chdir("flight_dataset")
    os.system("python create_flight_dataset.py")
    os.chdir("..")
    
    # 4. 检查飞行数据集
    flight_samples = check_flight_dataset()
    
    # 5. 对比分析
    print("\n" + "="*50)
    print("对比分析")
    print("="*50)
    
    if evtol_segments and flight_samples:
        print(f"样本数量对比:")
        print(f"  EVTOL段数: {evtol_segments}")
        print(f"  飞行样本数: {flight_samples}")
        
        if evtol_segments == flight_samples:
            print("  ✅ 样本数量一致")
        else:
            print(f"  ❌ 样本数量不一致，差异: {abs(evtol_segments - flight_samples)}")
    
    print("\n建议:")
    print("- 如果异常比例合理(30%-40%)，运行测试: cd analysis && python test_datasets.py")
    print("- 进行可视化检查: cd analysis && python visualize_datasets.py")

if __name__ == "__main__":
    # 确保在comp目录下运行
    if not Path("evtol_dataset").exists() or not Path("flight_dataset").exists():
        print("请在comp目录下运行此脚本")
        exit(1)
    
    main()