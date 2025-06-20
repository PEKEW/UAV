#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os

# 添加项目路径
import sys
sys.path.append('src')

from models.model import BatteryAnomalyLSTM
from data.dataset import get_battery_dataset

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, 
                device='cuda', save_path='battery_anomaly_model.pth'):
    """训练电池异常检测模型"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    patience=5, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            data, labels = data.to(device), labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs['logits'], labels)
            
            # 如果模型支持不确定性估计，添加不确定性损失
            if model.use_uncertainty:
                uncertainty_outputs = model(data, return_uncertainty=True)
                # 简单的不确定性正则化
                uncertainty_loss = torch.mean(uncertainty_outputs['uncertainty'])
                loss += 0.1 * uncertainty_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs['prediction'] == labels).sum().item()
            train_total += labels.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                data, labels = data.to(device), labels.to(device).squeeze()
                
                outputs = model(data)
                loss = criterion(outputs['logits'], labels)
                
                val_loss += loss.item()
                val_correct += (outputs['prediction'] == labels).sum().item()
                val_total += labels.size(0)
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, save_path)
            print(f'  新的最佳模型已保存: {save_path}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_confidence = []
    all_uncertainty = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Evaluating'):
            data, labels = data.to(device), labels.to(device).squeeze()
            
            # 获取预测结果和置信度
            confidence_results = model.predict_with_confidence(data)
            
            all_predictions.extend(confidence_results['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(confidence_results['probabilities'].cpu().numpy())
            all_confidence.extend(confidence_results['confidence'].cpu().numpy())
            
            if confidence_results['uncertainty'] is not None:
                all_uncertainty.extend(confidence_results['uncertainty'].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_confidence = np.array(all_confidence)
    
    # 计算指标
    accuracy = (all_predictions == all_labels).mean()
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['正常', '异常']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # ROC曲线
    roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"平均置信度: {np.mean(all_confidence):.4f}")
    
    if all_uncertainty:
        print(f"平均不确定性: {np.mean(all_uncertainty):.4f}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confidence': all_confidence,
        'uncertainty': all_uncertainty if all_uncertainty else None,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_results(training_history, evaluation_results):
    """绘制训练和评估结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 训练损失
    axes[0, 0].plot(training_history['train_losses'], label='Training Loss')
    axes[0, 0].plot(training_history['val_losses'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练准确率
    axes[0, 1].plot(training_history['train_accuracies'], label='Training Accuracy')
    axes[0, 1].plot(training_history['val_accuracies'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 混淆矩阵
    sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    
    # ROC曲线
    axes[1, 0].plot(evaluation_results['fpr'], evaluation_results['tpr'], 
                    label=f'ROC Curve (AUC = {evaluation_results["roc_auc"]:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 置信度分布
    axes[1, 1].hist(evaluation_results['confidence'], bins=50, alpha=0.7, 
                    color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Confidence Distribution')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 不确定性分布（如果有）
    if evaluation_results['uncertainty'] is not None:
        axes[1, 2].hist(evaluation_results['uncertainty'], bins=50, alpha=0.7, 
                        color='orange', edgecolor='black')
        axes[1, 2].set_title('Uncertainty Distribution')
        axes[1, 2].set_xlabel('Uncertainty')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Uncertainty Data', 
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Uncertainty Distribution')
    
    plt.tight_layout()
    plt.savefig('battery_anomaly_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_path = "Datasets/test"
    
    # 获取数据集
    print("加载数据集...")
    train_dataset, val_dataset, test_dataset = get_battery_dataset(
        data_path, 
        sequence_length=50,
        train_ratio=0.7,
        val_ratio=0.15,
        classification_mode=True
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    model = BatteryAnomalyLSTM(
        input_size=7,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.2,
        use_uncertainty=True
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("开始训练...")
    training_history = train_model(
        model, train_loader, val_loader,
        num_epochs=50,
        lr=0.001,
        device=device,
        save_path='battery_anomaly_model.pth'
    )
    
    # 加载最佳模型
    print("加载最佳模型进行评估...")
    checkpoint = torch.load('battery_anomaly_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    print("评估模型...")
    evaluation_results = evaluate_model(model, test_loader, device)
    
    # 绘制结果
    print("生成结果图表...")
    plot_results(training_history, evaluation_results)
    
    # 保存结果
    results_df = pd.DataFrame({
        'predictions': evaluation_results['predictions'],
        'labels': evaluation_results['labels'],
        'confidence': evaluation_results['confidence'],
        'normal_prob': evaluation_results['probabilities'][:, 0],
        'anomaly_prob': evaluation_results['probabilities'][:, 1]
    })
    
    if evaluation_results['uncertainty'] is not None:
        results_df['uncertainty'] = evaluation_results['uncertainty']
    
    results_df.to_csv('battery_anomaly_results.csv', index=False)
    print("结果已保存为: battery_anomaly_results.csv")
    
    print("\n训练和评估完成！")

if __name__ == "__main__":
    main()