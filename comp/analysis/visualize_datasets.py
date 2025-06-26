#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DatasetVisualizer:
    """数据集可视化检查类"""
    
    def __init__(self):
        self.evtol_dataset_path = Path("../evtol_dataset/evtol_anomaly_dataset.csv")
        self.evtol_segment_path = Path("../evtol_dataset/processed/segment_info.csv")
        self.flight_dataset_path = Path("../flight_dataset/flight_anomaly_dataset.csv")
        self.flight_sample_path = Path("../flight_dataset/sample_info.csv")
        
    def load_datasets(self):
        """加载所有数据集"""
        datasets = {}
        
        if self.evtol_dataset_path.exists():
            datasets['evtol_data'] = pd.read_csv(self.evtol_dataset_path)
            print(f"加载EVTOL数据集: {len(datasets['evtol_data'])} 行")
        
        if self.evtol_segment_path.exists():
            datasets['evtol_segments'] = pd.read_csv(self.evtol_segment_path)
            print(f"加载EVTOL段信息: {len(datasets['evtol_segments'])} 段")
            
        if self.flight_dataset_path.exists():
            datasets['flight_data'] = pd.read_csv(self.flight_dataset_path)
            print(f"加载飞行数据集: {len(datasets['flight_data'])} 行")
            
        if self.flight_sample_path.exists():
            datasets['flight_samples'] = pd.read_csv(self.flight_sample_path)
            print(f"加载飞行样本信息: {len(datasets['flight_samples'])} 样本")
            
        return datasets
    
    def visualize_evtol_dataset(self, datasets):
        """可视化EVTOL数据集"""
        if 'evtol_data' not in datasets or 'evtol_segments' not in datasets:
            print("EVTOL数据集不完整，跳过可视化")
            return
            
        evtol_data = datasets['evtol_data']
        evtol_segments = datasets['evtol_segments']
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 段级异常统计
        ax1 = fig.add_subplot(3, 4, 1)
        segment_labels = evtol_segments['label'].value_counts()
        colors = ['lightblue', 'lightcoral']
        ax1.pie(segment_labels.values, labels=['正常段', '异常段'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('EVTOL数据段异常分布')
        
        # 2. 异常持续时间分布
        ax2 = fig.add_subplot(3, 4, 2)
        anomaly_durations = evtol_segments[evtol_segments['anomaly_duration'] > 0]['anomaly_duration']
        ax2.hist(anomaly_durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(x=15, color='red', linestyle='--', label='异常阈值(15s)')
        ax2.set_xlabel('异常持续时间 (秒)')
        ax2.set_ylabel('频次')
        ax2.set_title('异常持续时间分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 点级异常统计
        ax3 = fig.add_subplot(3, 4, 3)
        point_labels = evtol_data['label'].value_counts()
        ax3.pie(point_labels.values, labels=['正常点', '异常点'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax3.set_title('EVTOL数据点异常分布')
        
        # 4. 文件分布
        ax4 = fig.add_subplot(3, 4, 4)
        if 'file_id' in evtol_segments.columns:
            file_counts = evtol_segments['file_id'].value_counts()
            ax4.bar(range(len(file_counts)), file_counts.values, color='skyblue')
            ax4.set_xlabel('文件ID')
            ax4.set_ylabel('段数量')
            ax4.set_title('各文件的段数量分布')
            ax4.set_xticks(range(len(file_counts)))
            ax4.set_xticklabels(file_counts.index, rotation=45)
        
        # 5-8. 电池特征可视化
        battery_features = ['Ecell_V', 'I_mA', 'Temperature__C', 'QCharge_mA_h']
        available_features = [f for f in battery_features if f in evtol_data.columns]
        
        for i, feature in enumerate(available_features[:4]):
            ax = fig.add_subplot(3, 4, 5 + i)
            
            normal_data = evtol_data[evtol_data['label'] == 0][feature]
            anomaly_data = evtol_data[evtol_data['label'] == 1][feature]
            
            ax.hist(normal_data, bins=50, alpha=0.6, label='正常', color='blue', density=True)
            ax.hist(anomaly_data, bins=50, alpha=0.6, label='异常', color='red', density=True)
            ax.set_xlabel(feature)
            ax.set_ylabel('密度')
            ax.set_title(f'{feature} 分布对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 9. 异常判定逻辑验证
        ax9 = fig.add_subplot(3, 4, 9)
        scatter_x = evtol_segments['anomaly_duration']
        scatter_y = evtol_segments['label']
        scatter_colors = ['blue' if d < 15 else 'red' for d in scatter_x]
        ax9.scatter(scatter_x, scatter_y, c=scatter_colors, alpha=0.6)
        ax9.axvline(x=15, color='red', linestyle='--', label='异常阈值(15s)')
        ax9.set_xlabel('异常持续时间 (秒)')
        ax9.set_ylabel('段标签')
        ax9.set_title('异常判定逻辑验证')
        ax9.set_yticks([0, 1])
        ax9.set_yticklabels(['正常', '异常'])
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. 时间窗口检查
        ax10 = fig.add_subplot(3, 4, 10)
        window_durations = evtol_segments['end_time'] - evtol_segments['start_time']
        ax10.hist(window_durations, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax10.axvline(x=30, color='red', linestyle='--', label='目标窗口(30s)')
        ax10.set_xlabel('时间窗口长度 (秒)')
        ax10.set_ylabel('频次')
        ax10.set_title('时间窗口长度分布')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evtol_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_flight_dataset(self, datasets):
        """可视化飞行数据集"""
        if 'flight_data' not in datasets or 'flight_samples' not in datasets:
            print("飞行数据集不完整，跳过可视化")
            return
            
        flight_data = datasets['flight_data']
        flight_samples = datasets['flight_samples']
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 样本级异常统计
        ax1 = fig.add_subplot(3, 4, 1)
        sample_labels = flight_samples['sample_label'].value_counts()
        colors = ['lightblue', 'lightcoral']
        ax1.pie(sample_labels.values, labels=['正常样本', '异常样本'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('飞行数据样本异常分布')
        
        # 2. 异常比例分布
        ax2 = fig.add_subplot(3, 4, 2)
        anomaly_ratios = flight_samples[flight_samples['anomaly_ratio'] > 0]['anomaly_ratio']
        ax2.hist(anomaly_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', label='异常阈值(50%)')
        ax2.set_xlabel('样本内异常比例')
        ax2.set_ylabel('频次')
        ax2.set_title('样本异常比例分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 点级异常统计
        ax3 = fig.add_subplot(3, 4, 3)
        point_labels = flight_data['label'].value_counts()
        ax3.pie(point_labels.values, labels=['正常点', '异常点'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax3.set_title('飞行数据点异常分布')
        
        # 4. 3D飞行轨迹
        ax4 = fig.add_subplot(3, 4, 4, projection='3d')
        normal_flight = flight_data[flight_data['label'] == 0]
        anomaly_flight = flight_data[flight_data['label'] == 1]
        
        # 采样显示（避免数据点过多）
        if len(normal_flight) > 5000:
            normal_sample = normal_flight.sample(n=5000)
        else:
            normal_sample = normal_flight
            
        if len(anomaly_flight) > 2000:
            anomaly_sample = anomaly_flight.sample(n=2000)
        else:
            anomaly_sample = anomaly_flight
        
        ax4.scatter(normal_sample['x'], normal_sample['y'], normal_sample['z'], 
                   c='blue', alpha=0.3, s=1, label='正常')
        ax4.scatter(anomaly_sample['x'], anomaly_sample['y'], anomaly_sample['z'], 
                   c='red', alpha=0.6, s=2, label='异常')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D飞行轨迹')
        ax4.legend()
        
        # 5-8. 飞行特征可视化
        flight_features = ['roll', 'pitch', 'yaw', 'velocity']
        
        for i, feature in enumerate(flight_features):
            ax = fig.add_subplot(3, 4, 5 + i)
            
            normal_data = flight_data[flight_data['label'] == 0][feature]
            anomaly_data = flight_data[flight_data['label'] == 1][feature]
            
            ax.hist(normal_data, bins=50, alpha=0.6, label='正常', color='blue', density=True)
            ax.hist(anomaly_data, bins=50, alpha=0.6, label='异常', color='red', density=True)
            ax.set_xlabel(feature)
            ax.set_ylabel('密度')
            ax.set_title(f'{feature} 分布对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 9. 异常判定逻辑验证
        ax9 = fig.add_subplot(3, 4, 9)
        scatter_x = flight_samples['anomaly_ratio']
        scatter_y = flight_samples['sample_label']
        scatter_colors = ['blue' if r < 0.5 else 'red' for r in scatter_x]
        ax9.scatter(scatter_x, scatter_y, c=scatter_colors, alpha=0.6)
        ax9.axvline(x=0.5, color='red', linestyle='--', label='异常阈值(50%)')
        ax9.set_xlabel('异常比例')
        ax9.set_ylabel('样本标签')
        ax9.set_title('异常判定逻辑验证')
        ax9.set_yticks([0, 1])
        ax9.set_yticklabels(['正常', '异常'])
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. 时间序列示例
        ax10 = fig.add_subplot(3, 4, 10)
        # 选择一个异常样本进行展示
        anomaly_sample_data = flight_data[flight_data['sample_label'] == 1]
        if len(anomaly_sample_data) > 0:
            sample_id = anomaly_sample_data['sample_id'].iloc[0]
            sample_data = flight_data[flight_data['sample_id'] == sample_id].sort_values('time')
            
            ax10.plot(sample_data['time'], sample_data['roll'], 'b-', alpha=0.7, label='Roll')
            ax10.plot(sample_data['time'], sample_data['pitch'], 'g-', alpha=0.7, label='Pitch')
            ax10.plot(sample_data['time'], sample_data['yaw'], 'm-', alpha=0.7, label='Yaw')
            
            # 标记异常区域
            anomaly_mask = sample_data['label'] == 1
            if anomaly_mask.any():
                ax10.fill_between(sample_data['time'], -50, 50, where=anomaly_mask, 
                                 color='red', alpha=0.2, label='异常区域')
            
            ax10.set_xlabel('时间 (s)')
            ax10.set_ylabel('角度 (度)')
            ax10.set_title(f'异常样本时间序列示例 (样本{sample_id})')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('flight_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def compare_datasets(self, datasets):
        """对比两个数据集"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 样本数量对比
        if 'evtol_segments' in datasets and 'flight_samples' in datasets:
            evtol_count = len(datasets['evtol_segments'])
            flight_count = len(datasets['flight_samples'])
            
            axes[0, 0].bar(['EVTOL', 'Flight'], [evtol_count, flight_count], 
                          color=['lightblue', 'lightgreen'])
            axes[0, 0].set_ylabel('样本数量')
            axes[0, 0].set_title('数据集样本数量对比')
            
            for i, v in enumerate([evtol_count, flight_count]):
                axes[0, 0].text(i, v + max(evtol_count, flight_count) * 0.01, 
                               str(v), ha='center', va='bottom')
        
        # 2. 异常比例对比
        if 'evtol_segments' in datasets and 'flight_samples' in datasets:
            evtol_anomaly_ratio = len(datasets['evtol_segments'][datasets['evtol_segments']['label'] == 1]) / len(datasets['evtol_segments'])
            flight_anomaly_ratio = len(datasets['flight_samples'][datasets['flight_samples']['sample_label'] == 1]) / len(datasets['flight_samples'])
            
            axes[0, 1].bar(['EVTOL', 'Flight'], [evtol_anomaly_ratio, flight_anomaly_ratio], 
                          color=['lightcoral', 'lightsalmon'])
            axes[0, 1].set_ylabel('异常比例')
            axes[0, 1].set_title('异常样本比例对比')
            axes[0, 1].set_ylim(0, 0.5)
            
            for i, v in enumerate([evtol_anomaly_ratio, flight_anomaly_ratio]):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. 数据点数量对比
        if 'evtol_data' in datasets and 'flight_data' in datasets:
            evtol_points = len(datasets['evtol_data'])
            flight_points = len(datasets['flight_data'])
            
            axes[0, 2].bar(['EVTOL', 'Flight'], [evtol_points, flight_points], 
                          color=['lightsteelblue', 'lightseagreen'])
            axes[0, 2].set_ylabel('数据点数量')
            axes[0, 2].set_title('数据集数据点数量对比')
            
            for i, v in enumerate([evtol_points, flight_points]):
                axes[0, 2].text(i, v + max(evtol_points, flight_points) * 0.01, 
                               str(v), ha='center', va='bottom')
        
        # 4. 异常持续时间/比例分布对比
        if 'evtol_segments' in datasets and 'flight_samples' in datasets:
            axes[1, 0].hist(datasets['evtol_segments'][datasets['evtol_segments']['anomaly_duration'] > 0]['anomaly_duration'], 
                           bins=15, alpha=0.6, label='EVTOL异常持续时间(s)', color='blue', density=True)
            axes[1, 0].hist(datasets['flight_samples'][datasets['flight_samples']['anomaly_ratio'] > 0]['anomaly_ratio'] * 30, 
                           bins=15, alpha=0.6, label='Flight异常持续时间(s)', color='red', density=True)
            axes[1, 0].set_xlabel('异常持续时间 (秒)')
            axes[1, 0].set_ylabel('密度')
            axes[1, 0].set_title('异常持续时间分布对比')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 特征数量对比
        if 'evtol_data' in datasets and 'flight_data' in datasets:
            evtol_features = [col for col in datasets['evtol_data'].columns 
                             if col not in ['label', 'segment_id', 'file_id', 'is_anomaly', 'anomaly_duration']]
            flight_features = [col for col in datasets['flight_data'].columns 
                              if col not in ['label', 'sample_id', 'sample_label', 'anomaly_ratio', 'time']]
            
            axes[1, 1].bar(['EVTOL', 'Flight'], [len(evtol_features), len(flight_features)], 
                          color=['wheat', 'plum'])
            axes[1, 1].set_ylabel('特征数量')
            axes[1, 1].set_title('数据集特征数量对比')
            
            for i, v in enumerate([len(evtol_features), len(flight_features)]):
                axes[1, 1].text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # 6. 总结信息
        axes[1, 2].axis('off')
        summary_text = "数据集对比总结:\n\n"
        
        if 'evtol_segments' in datasets and 'flight_samples' in datasets:
            evtol_count = len(datasets['evtol_segments'])
            flight_count = len(datasets['flight_samples'])
            summary_text += f"样本数量: EVTOL={evtol_count}, Flight={flight_count}\n"
            
            evtol_anomaly_ratio = len(datasets['evtol_segments'][datasets['evtol_segments']['label'] == 1]) / len(datasets['evtol_segments'])
            flight_anomaly_ratio = len(datasets['flight_samples'][datasets['flight_samples']['sample_label'] == 1]) / len(datasets['flight_samples'])
            summary_text += f"异常比例: EVTOL={evtol_anomaly_ratio:.3f}, Flight={flight_anomaly_ratio:.3f}\n"
            
            ratio_diff = abs(evtol_anomaly_ratio - flight_anomaly_ratio)
            if ratio_diff < 0.05:
                summary_text += "✓ 异常比例一致性: 良好\n"
            else:
                summary_text += "⚠ 异常比例一致性: 需要调整\n"
            
            count_diff = abs(evtol_count - flight_count)
            if count_diff == 0:
                summary_text += "✓ 样本数量一致性: 完美\n"
            elif count_diff <= evtol_count * 0.05:
                summary_text += "✓ 样本数量一致性: 良好\n"
            else:
                summary_text += "⚠ 样本数量一致性: 需要调整\n"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self, datasets):
        """生成总结报告"""
        report = []
        report.append("="*60)
        report.append("数据集可视化检查报告")
        report.append("="*60)
        
        # EVTOL数据集分析
        if 'evtol_data' in datasets and 'evtol_segments' in datasets:
            evtol_data = datasets['evtol_data']
            evtol_segments = datasets['evtol_segments']
            
            report.append("\n1. EVTOL数据集分析:")
            report.append(f"   - 总段数: {len(evtol_segments)}")
            report.append(f"   - 总数据点: {len(evtol_data)}")
            report.append(f"   - 异常段数: {len(evtol_segments[evtol_segments['label'] == 1])}")
            report.append(f"   - 异常段比例: {len(evtol_segments[evtol_segments['label'] == 1])/len(evtol_segments)*100:.1f}%")
            report.append(f"   - 异常数据点: {len(evtol_data[evtol_data['label'] == 1])}")
            report.append(f"   - 异常点比例: {len(evtol_data[evtol_data['label'] == 1])/len(evtol_data)*100:.1f}%")
            
            # 检查异常判定逻辑
            logic_errors = 0
            for _, row in evtol_segments.iterrows():
                if row['anomaly_duration'] >= 15 and row['label'] != 1:
                    logic_errors += 1
                elif row['anomaly_duration'] < 15 and row['anomaly_duration'] > 0 and row['label'] != 0:
                    logic_errors += 1
            
            if logic_errors == 0:
                report.append("   ✓ 异常判定逻辑: 正确")
            else:
                report.append(f"   ⚠ 异常判定逻辑: {logic_errors}个错误")
        
        # 飞行数据集分析
        if 'flight_data' in datasets and 'flight_samples' in datasets:
            flight_data = datasets['flight_data']
            flight_samples = datasets['flight_samples']
            
            report.append("\n2. 飞行数据集分析:")
            report.append(f"   - 总样本数: {len(flight_samples)}")
            report.append(f"   - 总数据点: {len(flight_data)}")
            report.append(f"   - 异常样本数: {len(flight_samples[flight_samples['sample_label'] == 1])}")
            report.append(f"   - 异常样本比例: {len(flight_samples[flight_samples['sample_label'] == 1])/len(flight_samples)*100:.1f}%")
            report.append(f"   - 异常数据点: {len(flight_data[flight_data['label'] == 1])}")
            report.append(f"   - 异常点比例: {len(flight_data[flight_data['label'] == 1])/len(flight_data)*100:.1f}%")
            
            # 检查异常判定逻辑
            logic_errors = 0
            for _, row in flight_samples.iterrows():
                if row['anomaly_ratio'] >= 0.5 and row['sample_label'] != 1:
                    logic_errors += 1
                elif row['anomaly_ratio'] < 0.5 and row['sample_label'] != 0:
                    logic_errors += 1
            
            if logic_errors == 0:
                report.append("   ✓ 异常判定逻辑: 正确")
            else:
                report.append(f"   ⚠ 异常判定逻辑: {logic_errors}个错误")
        
        # 数据集对比
        if all(key in datasets for key in ['evtol_segments', 'flight_samples']):
            report.append("\n3. 数据集对比:")
            evtol_count = len(datasets['evtol_segments'])
            flight_count = len(datasets['flight_samples'])
            
            report.append(f"   - 样本数量对比: EVTOL={evtol_count}, Flight={flight_count}")
            
            if evtol_count == flight_count:
                report.append("   ✓ 样本数量: 完全一致")
            else:
                report.append(f"   ⚠ 样本数量: 差异{abs(evtol_count - flight_count)}")
            
            evtol_anomaly_ratio = len(datasets['evtol_segments'][datasets['evtol_segments']['label'] == 1]) / len(datasets['evtol_segments'])
            flight_anomaly_ratio = len(datasets['flight_samples'][datasets['flight_samples']['sample_label'] == 1]) / len(datasets['flight_samples'])
            
            report.append(f"   - 异常比例对比: EVTOL={evtol_anomaly_ratio:.3f}, Flight={flight_anomaly_ratio:.3f}")
            
            ratio_diff = abs(evtol_anomaly_ratio - flight_anomaly_ratio)
            if ratio_diff < 0.05:
                report.append("   ✓ 异常比例: 一致性良好")
            else:
                report.append(f"   ⚠ 异常比例: 差异{ratio_diff:.3f}")
        
        report.append("\n" + "="*60)
        
        # 保存报告
        with open('dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 打印报告
        for line in report:
            print(line)
            
        return report

def main():
    """主函数"""
    visualizer = DatasetVisualizer()
    
    print("开始数据集可视化检查...")
    
    # 加载数据集
    datasets = visualizer.load_datasets()
    
    if not datasets:
        print("未找到任何数据集文件，请先运行数据生成脚本")
        return
    
    # 可视化EVTOL数据集
    print("\n生成EVTOL数据集可视化...")
    visualizer.visualize_evtol_dataset(datasets)
    
    # 可视化飞行数据集
    print("\n生成飞行数据集可视化...")
    visualizer.visualize_flight_dataset(datasets)
    
    # 对比数据集
    print("\n生成数据集对比可视化...")
    visualizer.compare_datasets(datasets)
    
    # 生成总结报告
    print("\n生成总结报告...")
    visualizer.generate_summary_report(datasets)
    
    print("\n可视化检查完成！")
    print("生成的文件:")
    print("- evtol_dataset_analysis.png")
    print("- flight_dataset_analysis.png") 
    print("- dataset_comparison.png")
    print("- dataset_analysis_report.txt")

if __name__ == "__main__":
    main()