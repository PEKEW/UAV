import pandas as pd
import numpy as np
import h5py
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import gc
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVTo3DConverter:
    """CSV到3D数据转换器"""
    
    def __init__(self, sequence_length: int = 30, chunk_size: int = 10000):
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        
        # 电池数据特征列
        self.battery_features = [
            'Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h',
            'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C'
        ]
        
        # 飞行数据特征列
        self.flight_features = [
            'x', 'y', 'z', 'roll', 'pitch', 'yaw', 
            'velocity', 'acceleration', 'altitude'
        ]
    
    def detect_data_type(self, df: pd.DataFrame) -> str:
        """检测数据类型"""
        if all(col in df.columns for col in self.battery_features[:3]):
            return 'battery'
        elif all(col in df.columns for col in self.flight_features[:3]):
            return 'flight'
        else:
            raise ValueError("无法检测数据类型，请检查特征列")
    
    def get_feature_columns(self, data_type: str, df: pd.DataFrame) -> list:
        """获取特征列"""
        if data_type == 'battery':
            return [col for col in self.battery_features if col in df.columns]
        else:
            return [col for col in self.flight_features if col in df.columns]
    
    def load_data_efficiently(self, csv_path: str) -> pd.DataFrame:
        """高效加载CSV数据"""
        logger.info(f"加载数据: {csv_path}")
        
        # 先读取前几行来检测数据类型
        sample_df = pd.read_csv(csv_path, nrows=5)
        data_type = self.detect_data_type(sample_df)
        logger.info(f"检测到数据类型: {data_type}")
        
        # 使用更高效的数据类型
        dtype_dict = {
            'sample_id': 'int32',
            'file_id': 'category',
            'sample_label': 'int8',
            'label': 'int8',
            'is_anomaly': 'int8',
            'anomaly_duration': 'float32',
            'time_s': 'float32',
            'time': 'float32'
        }
        
        # 根据数据类型设置特征列的数据类型
        if data_type == 'battery':
            for feature in self.battery_features:
                dtype_dict[feature] = 'float32'
        else:  # flight
            for feature in self.flight_features:
                dtype_dict[feature] = 'float32'
        
        # 只读取必要的列（动态检测可用的列）
        usecols = ['sample_id']
        
        # 检测标签列
        label_columns = ['sample_label', 'label', 'is_anomaly']
        available_label_cols = [col for col in label_columns if col in sample_df.columns]
        if available_label_cols:
            usecols.extend(available_label_cols)
        
        # 添加特征列
        if data_type == 'battery':
            for feature in self.battery_features:
                if feature in sample_df.columns:
                    usecols.append(feature)
        else:  # flight
            for feature in self.flight_features:
                if feature in sample_df.columns:
                    usecols.append(feature)
        
        # 如果存在时间列，也读取
        if 'time_s' in sample_df.columns:
            usecols.append('time_s')
        elif 'time' in sample_df.columns:
            usecols.append('time')
        
        # 如果存在其他元数据列，也读取
        meta_columns = ['file_id', 'anomaly_duration']
        for col in meta_columns:
            if col in sample_df.columns:
                usecols.append(col)
        
        logger.info(f"将读取的列: {usecols}")
        
        # 分块读取大文件
        chunks = []
        total_rows = sum(1 for _ in open(csv_path)) - 1  # 减去标题行
        
        logger.info(f"总行数: {total_rows:,}")
        
        for chunk in tqdm(pd.read_csv(csv_path, usecols=usecols, dtype=dtype_dict, 
                                     chunksize=self.chunk_size), 
                         desc="加载数据"):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"加载完成，数据形状: {df.shape}")
        
        return df
    
    def validate_sample_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证样本完整性"""
        logger.info("验证样本完整性...")
        
        # 统计每个样本的数据行数
        sample_counts = df['sample_id'].value_counts()
        
        # 找出完整的样本（有30行数据）
        complete_samples = sample_counts[sample_counts == self.sequence_length].index
        
        logger.info(f"完整样本数: {len(complete_samples):,} / {len(sample_counts):,}")
        
        # 只保留完整的样本
        df_filtered = df[df['sample_id'].isin(complete_samples)].copy()
        
        # 按sample_id和时间排序
        if 'time_s' in df_filtered.columns:
            df_filtered = df_filtered.sort_values(['sample_id', 'time_s'])
        elif 'time' in df_filtered.columns:
            df_filtered = df_filtered.sort_values(['sample_id', 'time'])
        else:
            df_filtered = df_filtered.sort_values('sample_id')
        
        return df_filtered
    
    def extract_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """提取样本标签"""
        logger.info("提取样本标签...")
        
        # 检测可用的标签列
        label_columns = ['sample_label', 'label', 'is_anomaly']
        available_label_cols = [col for col in label_columns if col in df.columns]
        
        if not available_label_cols:
            raise ValueError("未找到任何标签列，请检查数据格式")
        
        # 使用第一个可用的标签列
        label_col = available_label_cols[0]
        logger.info(f"使用标签列: {label_col}")
        
        # 获取每个样本的标签（取第一个，因为都是相同的）
        sample_labels = df.groupby('sample_id').agg({
            label_col: 'first'
        }).reset_index()
        
        labels = sample_labels[label_col].values.astype(np.int8)
        
        # 统计标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_stats = {
            'unique_labels': unique_labels.tolist(),
            'counts': counts.tolist(),
            'total_samples': len(labels),
            'anomaly_ratio': (labels == 1).mean(),
            'label_column_used': label_col
        }
        
        logger.info(f"标签分布: {dict(zip(unique_labels, counts))}")
        logger.info(f"异常比例: {label_stats['anomaly_ratio']:.3f}")
        
        return labels, label_stats
    
    def reshape_to_3d(self, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        """将数据重塑为3D格式"""
        logger.info("重塑数据为3D格式...")
        
        # 获取样本ID列表
        sample_ids = df['sample_id'].unique()
        n_samples = len(sample_ids)
        n_features = len(feature_columns)
        
        logger.info(f"样本数: {n_samples:,}, 特征数: {n_features}")
        
        # 创建3D数组 [samples, time_steps, features]
        data_3d = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        
        # 使用向量化操作填充数据
        for i, sample_id in enumerate(tqdm(sample_ids, desc="重塑数据")):
            sample_data = df[df['sample_id'] == sample_id]
            if len(sample_data) == self.sequence_length:
                data_3d[i] = sample_data[feature_columns].values
        
        logger.info(f"3D数据形状: {data_3d.shape}")
        return data_3d
    
    def save_to_h5(self, data_3d: np.ndarray, labels: np.ndarray, 
                   feature_columns: list, output_path: str, 
                   label_stats: Dict[str, Any], data_type: str):
        """保存为HDF5格式"""
        logger.info(f"保存到: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # 保存3D数据
            f.create_dataset('data', data=data_3d, compression='gzip', compression_opts=6)
            
            # 保存标签
            f.create_dataset('labels', data=labels, compression='gzip')
            
            # 保存元数据
            f.attrs['data_type'] = data_type
            f.attrs['sequence_length'] = self.sequence_length
            f.attrs['n_samples'] = len(labels)
            f.attrs['n_features'] = len(feature_columns)
            f.attrs['feature_columns'] = feature_columns
            f.attrs['label_stats'] = str(label_stats)
            
            # 保存特征列名
            f.create_dataset('feature_names', data=np.array(feature_columns, dtype='S20'))
        
        logger.info("保存完成")
    
    def save_to_npz(self, data_3d: np.ndarray, labels: np.ndarray, 
                    feature_columns: list, output_path: str, 
                    label_stats: Dict[str, Any], data_type: str):
        """保存为NPZ格式"""
        logger.info(f"保存到: {output_path}")
        
        np.savez_compressed(
            output_path,
            data=data_3d,
            labels=labels,
            feature_columns=feature_columns,
            data_type=data_type,
            sequence_length=self.sequence_length,
            label_stats=label_stats
        )
        
        logger.info("保存完成")
    
    def convert(self, csv_path: str, output_dir: str, format: str = 'h5') -> Dict[str, Any]:
        """主转换函数"""
        start_time = time.time()
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据（包含数据类型检测）
        self.raw_data = self.load_data_efficiently(csv_path)
        
        # 重新检测数据类型（从已加载的数据）
        data_type = self.detect_data_type(self.raw_data)
        
        # 获取特征列
        feature_columns = self.get_feature_columns(data_type, self.raw_data)
        logger.info(f"特征列: {feature_columns}")
        
        # 验证样本完整性
        self.raw_data = self.validate_sample_completeness(self.raw_data)
        
        # 提取标签
        labels, label_stats = self.extract_labels(self.raw_data)
        
        # 重塑为3D数据
        data_3d = self.reshape_to_3d(self.raw_data, feature_columns)
        
        # 保存数据
        base_name = Path(csv_path).stem
        if format == 'h5':
            output_path = output_dir / f"{base_name}_3d.h5"
            self.save_to_h5(data_3d, labels, feature_columns, str(output_path), 
                           label_stats, data_type)
        else:  # npz
            output_path = output_dir / f"{base_name}_3d.npz"
            self.save_to_npz(data_3d, labels, feature_columns, str(output_path), 
                            label_stats, data_type)
        
        # 统计信息（在删除变量之前计算）
        total_time = time.time() - start_time
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        result = {
            'input_file': csv_path,
            'output_file': str(output_path),
            'data_type': data_type,
            'data_shape': data_3d.shape,  # 在这里使用 data_3d
            'n_samples': len(labels),
            'n_features': len(feature_columns),
            'sequence_length': self.sequence_length,
            'label_stats': label_stats,
            'processing_time': total_time,
            'output_size_mb': file_size
        }
        
        # 清理内存（在创建结果字典之后）
        del self.raw_data, data_3d, labels
        gc.collect()
        
        logger.info(f"转换完成，耗时: {total_time:.2f}秒")
        logger.info(f"输出文件大小: {file_size:.2f} MB")
        
        return result


def load_3d_data(file_path: str) -> Dict[str, Any]:
    """加载3D数据文件"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.h5':
        with h5py.File(file_path, 'r') as f:
            data = {
                'data': f['data'][:],
                'labels': f['labels'][:],
                'feature_names': [name.decode('utf-8') for name in f['feature_names'][:]],
                'data_type': f.attrs['data_type'],
                'sequence_length': f.attrs['sequence_length'],
                'n_samples': f.attrs['n_samples'],
                'n_features': f.attrs['n_features']
            }
    else:  # .npz
        loaded = np.load(file_path, allow_pickle=True)
        data = {
            'data': loaded['data'],
            'labels': loaded['labels'],
            'feature_names': loaded['feature_columns'].tolist(),
            'data_type': str(loaded['data_type']),
            'sequence_length': int(loaded['sequence_length']),
            'n_samples': loaded['data'].shape[0],
            'n_features': loaded['data'].shape[2]
        }
    
    return data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将CSV文件转换为3D数据格式')
    parser.add_argument('--input-csv', default='flight_anomaly_dataset.csv', 
                    help='输入CSV文件路径 (默认: flight_anomaly_dataset.csv)')
    parser.add_argument('--output-dir', default='processed', 
                    help='输出目录 (默认: processed)')
    parser.add_argument('--format', choices=['h5', 'npz'], default='h5', 
                    help='输出格式 (默认: h5)')
    parser.add_argument('--sequence-length', type=int, default=30, 
                    help='序列长度 (默认: 30)')
    parser.add_argument('--chunk-size', type=int, default=10000, 
                    help='数据加载块大小 (默认: 10000)')
    parser.add_argument('--test-load', action='store_true', 
                    help='转换后测试加载数据')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = CSVTo3DConverter(
        sequence_length=args.sequence_length,
        chunk_size=args.chunk_size
    )
    
    # 执行转换
    result = converter.convert(args.input_csv, args.output_dir, args.format)
    
    # 打印结果
    print("\n" + "="*50)
    print("转换结果:")
    print(f"输入文件: {result['input_file']}")
    print(f"输出文件: {result['output_file']}")
    print(f"数据类型: {result['data_type']}")
    print(f"数据形状: {result['data_shape']}")
    print(f"样本数: {result['n_samples']:,}")
    print(f"特征数: {result['n_features']}")
    print(f"序列长度: {result['sequence_length']}")
    print(f"异常比例: {result['label_stats']['anomaly_ratio']:.3f}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"输出大小: {result['output_size_mb']:.2f} MB")
    print("="*50)
    
    # 测试加载
    if args.test_load:
        print("\n测试加载数据...")
        loaded_data = load_3d_data(result['output_file'])
        print(f"加载成功: {loaded_data['data'].shape}")
        print(f"标签分布: {np.bincount(loaded_data['labels'])}")


if __name__ == '__main__':
    main() 