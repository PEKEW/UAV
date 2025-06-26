"""
评估指标模块
包含异常检测任务的各种评估指标
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


class AnomalyMetrics:
    """异常检测评估指标"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def compute_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算批次指标"""
        with torch.no_grad():
            # 转换为全精度浮点数以避免半精度兼容性问题
            outputs = outputs.float()
            targets = targets.float()
            
            # 转换为概率
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # 转换为numpy
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            metrics = {}
            
            # 基本分类指标
            metrics['accuracy'] = accuracy_score(targets_np, preds_np)
            metrics['precision'] = precision_score(targets_np, preds_np, average='binary', zero_division=0)
            metrics['recall'] = recall_score(targets_np, preds_np, average='binary', zero_division=0)
            metrics['f1'] = f1_score(targets_np, preds_np, average='binary', zero_division=0)
            
            # ROC-AUC (需要概率)
            if len(np.unique(targets_np)) > 1:  # 确保有两个类别
                try:
                    if probs_np.shape[1] == 2:  # 二分类
                        metrics['auc'] = roc_auc_score(targets_np, probs_np[:, 1])
                        metrics['ap'] = average_precision_score(targets_np, probs_np[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(targets_np, probs_np, multi_class='ovr', average='macro')
                except:
                    metrics['auc'] = 0.0
                    metrics['ap'] = 0.0
            else:
                metrics['auc'] = 0.0
                metrics['ap'] = 0.0
            
            return metrics
    
    def compute_epoch_metrics(self, all_outputs: torch.Tensor, all_targets: torch.Tensor) -> Dict[str, float]:
        """计算整个epoch的指标"""
        with torch.no_grad():
            # 转换为全精度浮点数以避免半精度兼容性问题
            all_outputs = all_outputs.float()
            all_targets = all_targets.float()
            
            # 转换为概率和预测
            probs = torch.softmax(all_outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # 转换为numpy
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = all_targets.cpu().numpy()
            
            metrics = {}
            
            # 基本分类指标
            metrics['accuracy'] = accuracy_score(targets_np, preds_np)
            metrics['precision'] = precision_score(targets_np, preds_np, average='binary', zero_division=0)
            metrics['recall'] = recall_score(targets_np, preds_np, average='binary', zero_division=0)
            metrics['f1'] = f1_score(targets_np, preds_np, average='binary', zero_division=0)
            
            # 类别特定指标
            metrics['precision_macro'] = precision_score(targets_np, preds_np, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(targets_np, preds_np, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(targets_np, preds_np, average='macro', zero_division=0)
            
            # ROC-AUC和PR-AUC
            if len(np.unique(targets_np)) > 1:
                try:
                    if probs_np.shape[1] == 2:  # 二分类
                        metrics['auc'] = roc_auc_score(targets_np, probs_np[:, 1])
                        metrics['ap'] = average_precision_score(targets_np, probs_np[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(targets_np, probs_np, multi_class='ovr', average='macro')
                        metrics['ap'] = average_precision_score(targets_np, probs_np, average='macro')
                except:
                    metrics['auc'] = 0.0
                    metrics['ap'] = 0.0
            else:
                metrics['auc'] = 0.0
                metrics['ap'] = 0.0
            
            # 混淆矩阵相关指标
            cm = confusion_matrix(targets_np, preds_np)
            if cm.shape == (2, 2):  # 二分类
                tn, fp, fn, tp = cm.ravel()
                
                # 特异性 (Specificity)
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # 敏感性 (Sensitivity) = Recall
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # 阳性预测值 (PPV) = Precision
                metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                # 阴性预测值 (NPV)
                metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                
                # False Positive Rate
                metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                # False Negative Rate
                metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                
                # Matthews Correlation Coefficient
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                if denominator > 0:
                    metrics['mcc'] = (tp * tn - fp * fn) / denominator
                else:
                    metrics['mcc'] = 0.0
                
                # Balanced Accuracy
                metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
            
            return metrics
    
    def compute_confusion_matrix(self, all_outputs: torch.Tensor, all_targets: torch.Tensor) -> np.ndarray:
        """计算混淆矩阵"""
        with torch.no_grad():
            # 转换为全精度浮点数
            all_outputs = all_outputs.float()
            all_targets = all_targets.float()
            
            preds = torch.argmax(torch.softmax(all_outputs, dim=-1), dim=-1)
            return confusion_matrix(all_targets.cpu().numpy(), preds.cpu().numpy())
    
    def get_classification_report(self, all_outputs: torch.Tensor, all_targets: torch.Tensor) -> str:
        """获取分类报告"""
        with torch.no_grad():
            # 转换为全精度浮点数
            all_outputs = all_outputs.float()
            all_targets = all_targets.float()
            
            preds = torch.argmax(torch.softmax(all_outputs, dim=-1), dim=-1)
            return classification_report(
                all_targets.cpu().numpy(), 
                preds.cpu().numpy(),
                target_names=['Normal', 'Anomaly'],
                digits=4
            )


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        self.best_metrics = {}
    
    def update(self, phase: str, epoch: int, metrics: Dict[str, float]):
        """更新指标"""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history[phase].append(metrics_with_epoch)
        
        # 更新最佳指标
        if phase not in self.best_metrics:
            self.best_metrics[phase] = {}
        
        for metric_name, value in metrics.items():
            if metric_name not in self.best_metrics[phase]:
                self.best_metrics[phase][metric_name] = {'value': value, 'epoch': epoch}
            else:
                # 对于大部分指标，值越大越好
                if metric_name in ['loss']:  # 损失越小越好
                    if value < self.best_metrics[phase][metric_name]['value']:
                        self.best_metrics[phase][metric_name] = {'value': value, 'epoch': epoch}
                else:  # 其他指标越大越好
                    if value > self.best_metrics[phase][metric_name]['value']:
                        self.best_metrics[phase][metric_name] = {'value': value, 'epoch': epoch}
    
    def get_best_metric(self, phase: str, metric_name: str) -> Tuple[float, int]:
        """获取最佳指标值和对应的epoch"""
        if phase in self.best_metrics and metric_name in self.best_metrics[phase]:
            best_info = self.best_metrics[phase][metric_name]
            return best_info['value'], best_info['epoch']
        return 0.0, 0
    
    def get_latest_metrics(self, phase: str) -> Dict[str, float]:
        """获取最新的指标"""
        if phase in self.metrics_history and self.metrics_history[phase]:
            return self.metrics_history[phase][-1]
        return {}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标总结"""
        summary = {}
        
        for phase in self.metrics_history:
            if self.metrics_history[phase]:
                latest = self.metrics_history[phase][-1]
                summary[phase] = {
                    'latest': latest,
                    'best': self.best_metrics.get(phase, {})
                }
        
        return summary
    
    def plot_metrics(self, metrics_to_plot: List[str] = None, save_path: str = None):
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，无法绘制图表")
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = ['loss', 'accuracy', 'f1', 'auc']
        
        # 获取所有可用的指标
        all_metrics = set()
        for phase in self.metrics_history:
            for epoch_metrics in self.metrics_history[phase]:
                all_metrics.update(epoch_metrics.keys())
        all_metrics.discard('epoch')
        
        # 过滤要绘制的指标
        metrics_to_plot = [m for m in metrics_to_plot if m in all_metrics]
        
        if not metrics_to_plot:
            print("没有找到要绘制的指标")
            return
        
        # 创建子图
        num_metrics = len(metrics_to_plot)
        cols = min(2, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个指标
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            for phase in ['train', 'val']:
                if phase in self.metrics_history and self.metrics_history[phase]:
                    epochs = [m['epoch'] for m in self.metrics_history[phase]]
                    values = [m.get(metric, 0) for m in self.metrics_history[phase]]
                    
                    if values:
                        ax.plot(epochs, values, label=f'{phase}_{metric}', marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()


class ModelPerformanceAnalyzer:
    """模型性能分析器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def analyze_predictions(self, data_loader, class_names: List[str] = None) -> Dict[str, Any]:
        """分析模型预测结果"""
        if class_names is None:
            class_names = ['Normal', 'Anomaly']
        
        self.model.eval()
        all_outputs = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=-1)
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        
        # 计算指标
        metrics = AnomalyMetrics().compute_epoch_metrics(all_outputs, all_targets)
        
        # 混淆矩阵
        cm = AnomalyMetrics().compute_confusion_matrix(all_outputs, all_targets)
        
        # 分类报告
        report = AnomalyMetrics().get_classification_report(all_outputs, all_targets)
        
        # 预测置信度分析
        preds = torch.argmax(all_probs, dim=-1)
        confidence_scores = torch.max(all_probs, dim=-1)[0]
        
        confidence_analysis = {
            'mean_confidence': float(confidence_scores.mean()),
            'std_confidence': float(confidence_scores.std()),
            'correct_predictions_confidence': float(confidence_scores[preds == all_targets].mean()),
            'incorrect_predictions_confidence': float(confidence_scores[preds != all_targets].mean()) if (preds != all_targets).any() else 0.0
        }
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'confidence_analysis': confidence_analysis,
            'class_names': class_names
        }