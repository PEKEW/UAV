import numpy as np
import pandas as pd
import h5py
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import random
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualDatasetCreator:
    """创建可视化用的小型数据集"""
    
    def __init__(self, total_samples: int = 99, positive_ratio: float = 1/3):
        self.total_samples = total_samples
        self.positive_ratio = positive_ratio
        self.negative_samples = int(total_samples * (1 - positive_ratio))
        self.positive_samples = total_samples - self.negative_samples
        
        logger.info(f"目标样本数: {total_samples}")
        logger.info(f"正样本数: {self.positive_samples} ({positive_ratio:.1%})")
        logger.info(f"负样本数: {self.negative_samples} ({(1-positive_ratio):.1%})")
    
    def load_h5_dataset(self, h5_path: str) -> Dict[str, Any]:
        """加载H5数据集"""
        logger.info(f"加载数据集: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            data = {
                'data': f['data'][:],
                'labels': f['labels'][:],
                'feature_names': [name.decode('utf-8') for name in f['feature_names'][:]],
                'data_type': f.attrs['data_type'],
                'sequence_length': f.attrs['sequence_length'],
                'n_samples': f.attrs['n_samples'],
                'n_features': f.attrs['n_features']
            }
        
        logger.info(f"数据集加载完成:")
        logger.info(f"  数据形状: {data['data'].shape}")
        logger.info(f"  标签形状: {data['labels'].shape}")
        logger.info(f"  特征名称: {data['feature_names']}")
        logger.info(f"  数据类型: {data['data_type']}")
        
        return data
    
    def split_samples_by_label(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """按标签分割样本"""
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        logger.info(f"原始数据集:")
        logger.info(f"  正样本数: {len(positive_indices)}")
        logger.info(f"  负样本数: {len(negative_indices)}")
        
        return positive_indices, negative_indices
    
    def select_samples(self, positive_indices: np.ndarray, negative_indices: np.ndarray) -> np.ndarray:
        """选择指定数量的正负样本"""
        # 设置随机种子确保可重复性
        random.seed(42)
        np.random.seed(42)
        
        # 随机选择正样本
        if len(positive_indices) >= self.positive_samples:
            selected_positive = np.random.choice(positive_indices, self.positive_samples, replace=False)
        else:
            logger.warning(f"正样本数量不足，需要 {self.positive_samples}，实际只有 {len(positive_indices)}")
            selected_positive = positive_indices
        
        # 随机选择负样本
        if len(negative_indices) >= self.negative_samples:
            selected_negative = np.random.choice(negative_indices, self.negative_samples, replace=False)
        else:
            logger.warning(f"负样本数量不足，需要 {self.negative_samples}，实际只有 {len(negative_indices)}")
            selected_negative = negative_indices
        
        # 合并并打乱顺序
        selected_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(selected_indices)
        
        actual_positive = len(selected_positive)
        actual_negative = len(selected_negative)
        actual_total = len(selected_indices)
        
        logger.info(f"选择的样本:")
        logger.info(f"  正样本数: {actual_positive}")
        logger.info(f"  负样本数: {actual_negative}")
        logger.info(f"  总样本数: {actual_total}")
        logger.info(f"  实际正样本比例: {actual_positive/actual_total:.1%}")
        
        return selected_indices
    
    def create_visual_dataset(self, data: np.ndarray, labels: np.ndarray, 
                            feature_names: list, selected_indices: np.ndarray,
                            data_type: str, sequence_length: int) -> Dict[str, Any]:
        """创建可视化数据集"""
        # 提取选中的样本
        visual_data = data[selected_indices]
        visual_labels = labels[selected_indices]
        
        # 计算标签统计
        unique_labels, counts = np.unique(visual_labels, return_counts=True)
        label_stats = {
            'total_samples': len(visual_labels),
            'positive_samples': counts[1] if len(counts) > 1 else 0,
            'negative_samples': counts[0],
            'positive_ratio': counts[1] / len(visual_labels) if len(counts) > 1 else 0
        }
        
        logger.info(f"可视化数据集统计:")
        logger.info(f"  数据形状: {visual_data.shape}")
        logger.info(f"  标签分布: {dict(zip(unique_labels, counts))}")
        logger.info(f"  正样本比例: {label_stats['positive_ratio']:.1%}")
        
        return {
            'data': visual_data,
            'labels': visual_labels,
            'feature_names': feature_names,
            'data_type': data_type,
            'sequence_length': sequence_length,
            'label_stats': label_stats
        }
    
    def save_visual_dataset(self, visual_dataset: Dict[str, Any], output_path: str):
        """保存可视化数据集"""
        logger.info(f"保存可视化数据集到: {output_path}")
        
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # 保存3D数据
            f.create_dataset('data', data=visual_dataset['data'], compression='gzip', compression_opts=6)
            
            # 保存标签
            f.create_dataset('labels', data=visual_dataset['labels'], compression='gzip')
            
            # 保存元数据
            f.attrs['data_type'] = visual_dataset['data_type']
            f.attrs['sequence_length'] = visual_dataset['sequence_length']
            f.attrs['n_samples'] = len(visual_dataset['labels'])
            f.attrs['n_features'] = len(visual_dataset['feature_names'])
            f.attrs['feature_columns'] = visual_dataset['feature_names']
            f.attrs['label_stats'] = str(visual_dataset['label_stats'])
            
            # 保存特征列名
            f.create_dataset('feature_names', data=np.array(visual_dataset['feature_names'], dtype='S20'))
        
        # 计算文件大小
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"数据集已保存，文件大小: {file_size:.2f} MB")
    
    def create_from_h5(self, input_h5_path: str, output_h5_path: str) -> Dict[str, Any]:
        """从H5文件创建可视化数据集"""
        # 加载原始数据集
        original_data = self.load_h5_dataset(input_h5_path)
        
        # 按标签分割样本
        positive_indices, negative_indices = self.split_samples_by_label(
            original_data['data'], original_data['labels']
        )
        
        # 选择样本
        selected_indices = self.select_samples(positive_indices, negative_indices)
        
        # 创建可视化数据集
        visual_dataset = self.create_visual_dataset(
            original_data['data'],
            original_data['labels'],
            original_data['feature_names'],
            selected_indices,
            original_data['data_type'],
            original_data['sequence_length']
        )
        
        # 保存数据集
        self.save_visual_dataset(visual_dataset, output_h5_path)
        
        return {
            'input_file': input_h5_path,
            'output_file': output_h5_path,
            'data_shape': visual_dataset['data'].shape,
            'label_stats': visual_dataset['label_stats'],
            'data_type': visual_dataset['data_type'],
            'sequence_length': visual_dataset['sequence_length']
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从H5数据集创建可视化用的小型数据集')
    parser.add_argument('--input-h5', required=True,
                        help='输入H5文件路径')
    parser.add_argument('--output-h5', required=True,
                        help='输出H5文件路径')
    parser.add_argument('--total-samples', type=int, default=99,
                        help='总样本数 (默认: 99)')
    parser.add_argument('--positive-ratio', type=float, default=1/3,
                        help='正样本比例 (默认: 1/3)')
    
    args = parser.parse_args()
    
    # 创建数据集创建器
    creator = VisualDatasetCreator(
        total_samples=args.total_samples,
        positive_ratio=args.positive_ratio
    )
    
    # 创建可视化数据集
    result = creator.create_from_h5(args.input_h5, args.output_h5)
    
    # 打印结果
    print("\n" + "="*50)
    print("可视化数据集创建结果:")
    print(f"输入文件: {result['input_file']}")
    print(f"输出文件: {result['output_file']}")
    print(f"数据形状: {result['data_shape']}")
    print(f"数据类型: {result['data_type']}")
    print(f"序列长度: {result['sequence_length']}")
    print(f"样本统计:")
    print(f"  总样本数: {result['label_stats']['total_samples']}")
    print(f"  正样本数: {result['label_stats']['positive_samples']}")
    print(f"  负样本数: {result['label_stats']['negative_samples']}")
    print(f"  正样本比例: {result['label_stats']['positive_ratio']:.1%}")
    print("="*50)


if __name__ == '__main__':
    main() 