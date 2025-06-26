#!/usr/bin/env python3
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

class TestEVTOLDataset(unittest.TestCase):
    """EVTOL数据集测试类"""
    
    def setUp(self):
        self.evtol_dataset_path = Path("../evtol_dataset/evtol_anomaly_dataset.csv")
        self.evtol_segment_path = Path("../evtol_dataset/processed/segment_info.csv")
        
    def test_evtol_dataset_exists(self):
        """测试EVTOL数据集文件是否存在"""
        self.assertTrue(self.evtol_dataset_path.exists(), "EVTOL数据集文件不存在")
        
    def test_evtol_segment_info_exists(self):
        """测试EVTOL段信息文件是否存在"""
        self.assertTrue(self.evtol_segment_path.exists(), "EVTOL段信息文件不存在")
        
    def test_evtol_dataset_structure(self):
        """测试EVTOL数据集结构"""
        if not self.evtol_dataset_path.exists():
            self.skipTest("EVTOL数据集文件不存在")
            
        df = pd.read_csv(self.evtol_dataset_path)
        
        # 检查必要的列
        required_columns = ['label', 'segment_id', 'file_id']
        for col in required_columns:
            self.assertIn(col, df.columns, f"缺少必要列: {col}")
            
        # 检查标签值
        unique_labels = df['label'].unique()
        self.assertTrue(set(unique_labels).issubset({0, 1}), "标签值应该只包含0和1")
        
    def test_evtol_anomaly_ratio(self):
        """测试EVTOL数据集异常比例"""
        if not self.evtol_dataset_path.exists():
            self.skipTest("EVTOL数据集文件不存在")
            
        df = pd.read_csv(self.evtol_dataset_path)
        
        total_points = len(df)
        anomaly_points = len(df[df['label'] == 1])
        anomaly_ratio = anomaly_points / total_points
        
        # 检查异常比例是否在合理范围内 (目标1/3 ± 10%)
        self.assertGreater(anomaly_ratio, 0.23, "异常比例过低")
        self.assertLess(anomaly_ratio, 0.43, "异常比例过高")
        
    def test_evtol_segment_anomaly_logic(self):
        """测试EVTOL数据段异常逻辑"""
        if not self.evtol_segment_path.exists():
            self.skipTest("EVTOL段信息文件不存在")
            
        segment_df = pd.read_csv(self.evtol_segment_path)
        
        # 检查异常判定逻辑：异常时长≥15秒的段应该标记为异常
        for _, row in segment_df.iterrows():
            if row['anomaly_duration'] >= 15:
                self.assertEqual(row['label'], 1, 
                               f"段{row['segment_id']}异常时长{row['anomaly_duration']}秒≥15秒，应标记为异常")
            elif row['anomaly_duration'] < 15 and row['anomaly_duration'] > 0:
                self.assertEqual(row['label'], 0, 
                               f"段{row['segment_id']}异常时长{row['anomaly_duration']}秒<15秒，应标记为正常")
                
    def test_evtol_time_window(self):
        """测试EVTOL数据30秒时间窗口"""
        if not self.evtol_segment_path.exists():
            self.skipTest("EVTOL段信息文件不存在")
            
        segment_df = pd.read_csv(self.evtol_segment_path)
        
        for _, row in segment_df.iterrows():
            duration = row['end_time'] - row['start_time']
            self.assertAlmostEqual(duration, 30.0, delta=1.0, 
                                 msg=f"段{row['segment_id']}时长{duration}秒，应该接近30秒")

class TestFlightDataset(unittest.TestCase):
    """飞行数据集测试类"""
    
    def setUp(self):
        self.flight_dataset_path = Path("../flight_dataset/flight_anomaly_dataset.csv")
        self.flight_sample_path = Path("../flight_dataset/sample_info.csv")
        
    def test_flight_dataset_exists(self):
        """测试飞行数据集文件是否存在"""
        self.assertTrue(self.flight_dataset_path.exists(), "飞行数据集文件不存在")
        
    def test_flight_sample_info_exists(self):
        """测试飞行样本信息文件是否存在"""
        self.assertTrue(self.flight_sample_path.exists(), "飞行样本信息文件不存在")
        
    def test_flight_dataset_structure(self):
        """测试飞行数据集结构"""
        if not self.flight_dataset_path.exists():
            self.skipTest("飞行数据集文件不存在")
            
        df = pd.read_csv(self.flight_dataset_path)
        
        # 检查必要的列
        required_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                          'velocity', 'acceleration', 'altitude', 'label', 'sample_label']
        for col in required_columns:
            self.assertIn(col, df.columns, f"缺少必要列: {col}")
            
        # 检查标签值
        unique_labels = df['label'].unique()
        self.assertTrue(set(unique_labels).issubset({0, 1}), "标签值应该只包含0和1")
        
    def test_flight_anomaly_ratio(self):
        """测试飞行数据集异常比例"""
        if not self.flight_sample_path.exists():
            self.skipTest("飞行样本信息文件不存在")
            
        sample_df = pd.read_csv(self.flight_sample_path)
        
        total_samples = len(sample_df)
        anomaly_samples = len(sample_df[sample_df['sample_label'] == 1])
        anomaly_ratio = anomaly_samples / total_samples
        
        # 检查异常比例是否在合理范围内 (目标1/3 ± 10%)
        self.assertGreater(anomaly_ratio, 0.23, "异常样本比例过低")
        self.assertLess(anomaly_ratio, 0.43, "异常样本比例过高")
        
    def test_flight_sample_anomaly_logic(self):
        """测试飞行样本异常逻辑"""
        if not self.flight_sample_path.exists():
            self.skipTest("飞行样本信息文件不存在")
            
        sample_df = pd.read_csv(self.flight_sample_path)
        
        # 检查异常判定逻辑：异常比例≥0.5的样本应该标记为异常
        for _, row in sample_df.iterrows():
            if row['anomaly_ratio'] >= 0.5:
                self.assertEqual(row['sample_label'], 1, 
                               f"样本{row['sample_id']}异常比例{row['anomaly_ratio']:.2f}≥0.5，应标记为异常")
            else:
                self.assertEqual(row['sample_label'], 0, 
                               f"样本{row['sample_id']}异常比例{row['anomaly_ratio']:.2f}<0.5，应标记为正常")

class TestDatasetComparison(unittest.TestCase):
    """数据集对比测试类"""
    
    def setUp(self):
        self.evtol_segment_path = Path("../evtol_dataset/processed/segment_info.csv")
        self.flight_sample_path = Path("../flight_dataset/sample_info.csv")
        
    def test_sample_count_consistency(self):
        """测试样本数量一致性"""
        if not (self.evtol_segment_path.exists() and self.flight_sample_path.exists()):
            self.skipTest("数据集文件不完整")
            
        evtol_segments = pd.read_csv(self.evtol_segment_path)
        flight_samples = pd.read_csv(self.flight_sample_path)
        
        evtol_count = len(evtol_segments)
        flight_count = len(flight_samples)
        
        self.assertEqual(evtol_count, flight_count, 
                        f"EVTOL数据集样本数({evtol_count})与飞行数据集样本数({flight_count})不一致")
        
    def test_anomaly_ratio_consistency(self):
        """测试异常比例一致性"""
        if not (self.evtol_segment_path.exists() and self.flight_sample_path.exists()):
            self.skipTest("数据集文件不完整")
            
        evtol_segments = pd.read_csv(self.evtol_segment_path)
        flight_samples = pd.read_csv(self.flight_sample_path)
        
        evtol_anomaly_ratio = len(evtol_segments[evtol_segments['label'] == 1]) / len(evtol_segments)
        flight_anomaly_ratio = len(flight_samples[flight_samples['sample_label'] == 1]) / len(flight_samples)
        
        # 允许5%的差异
        self.assertAlmostEqual(evtol_anomaly_ratio, flight_anomaly_ratio, delta=0.05,
                              msg=f"EVTOL异常比例({evtol_anomaly_ratio:.3f})与飞行数据集异常比例({flight_anomaly_ratio:.3f})差异过大")

class TestDataQuality(unittest.TestCase):
    """数据质量测试类"""
    
    def test_evtol_data_quality(self):
        """测试EVTOL数据质量"""
        evtol_path = Path("../evtol_dataset/evtol_anomaly_dataset.csv")
        if not evtol_path.exists():
            self.skipTest("EVTOL数据集文件不存在")
            
        df = pd.read_csv(evtol_path)
        
        # 检查空值
        null_counts = df.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, f"数据集包含空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 检查无穷值
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        self.assertEqual(inf_counts.sum(), 0, f"数据集包含无穷值: {inf_counts[inf_counts > 0].to_dict()}")
        
    def test_flight_data_quality(self):
        """测试飞行数据质量"""
        flight_path = Path("../flight_dataset/flight_anomaly_dataset.csv")
        if not flight_path.exists():
            self.skipTest("飞行数据集文件不存在")
            
        df = pd.read_csv(flight_path)
        
        # 检查空值
        null_counts = df.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, f"数据集包含空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 检查无穷值
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        self.assertEqual(inf_counts.sum(), 0, f"数据集包含无穷值: {inf_counts[inf_counts > 0].to_dict()}")

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [TestEVTOLDataset, TestFlightDataset, 
                   TestDatasetComparison, TestDataQuality]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print(f"\n{'='*50}")
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].strip()}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)