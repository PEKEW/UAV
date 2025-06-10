import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
from src.models.model import LSTM
from src.data.dataset import EVTOLDataset
import json

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sys.path.append('src')

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path, config):
    print(f"Loading model from {model_path}...")
    model = LSTM(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def load_test_data(data_path, config):
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))
    file_ids = [f.stem for f in csv_files]
    dataset = EVTOLDataset(
        data_path=str(data_path),
        sequence_length=config['data']['sequence_length'],
        prediction_steps=config['data']['prediction_steps'],
        file_ids=file_ids
    )
    return dataset

def generate_predictions(model, dataset, device='cpu', batch_size=256, use_uncertainty=False):
    model.eval()
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    uncertainties = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc='Prediction Progress'):
            batch_x = batch_x.to(device)
            
            if use_uncertainty and hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                # 使用置信区间预测
                outputs = model.predict_with_confidence(batch_x, confidence_level=0.95)
                if isinstance(outputs, dict):
                    if model.use_multi_task:
                        pred_key = list(model.prediction_targets)[0]
                        pred = outputs[pred_key]['prediction'].cpu().numpy()
                        std = outputs[pred_key]['std'].cpu().numpy()
                    else:
                        pred = outputs['prediction'].cpu().numpy()
                        std = outputs['std'].cpu().numpy()
                    uncertainties.extend(std)
                else:
                    pred = outputs.cpu().numpy()
                    uncertainties.extend(np.zeros_like(pred))
            else:
                # 标准预测
                outputs = model(batch_x)
                if isinstance(outputs, dict):
                    pred_key = list(outputs.keys())[0]
                    if isinstance(outputs[pred_key], dict):
                        pred = outputs[pred_key]['prediction'].cpu().numpy()
                        std = outputs[pred_key].get('std', torch.zeros_like(outputs[pred_key]['prediction'])).cpu().numpy()
                        uncertainties.extend(std)
                    else:
                        pred = outputs[pred_key].cpu().numpy()
                        uncertainties.extend(np.zeros_like(pred))
                else:
                    pred = outputs.cpu().numpy()
                    uncertainties.extend(np.zeros_like(pred))
            
            if len(batch_y.shape) == 3:
                # TODO 这里默认第一个是电压
                target = batch_y[:, :, 0].cpu().numpy()
            else:
                target = batch_y.cpu().numpy()
            predictions.extend(pred)
            targets.extend(target)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    uncertainties = np.array(uncertainties)
    return predictions, targets, uncertainties

def calculate_metrics(predictions, targets):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    mae = mean_absolute_error(target_flat, pred_flat)
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_flat, pred_flat)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2
    }

def plot_predictions(predictions, targets, config, uncertainties=None, save_path='prediction_comparison.png'):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Battery Voltage Prediction Comparison Analysis', fontsize=16, fontweight='bold')
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    metrics = calculate_metrics(predictions, targets)
    n_points = min(1000, len(pred_flat))
    x_axis = np.arange(n_points)
    axes[0, 0].plot(x_axis, target_flat[:n_points], 'b-', label='True Values', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(x_axis, pred_flat[:n_points], 'r-', label='Predictions', alpha=0.7, linewidth=1.5)
    
    # 添加置信区间
    if uncertainties is not None:
        uncert_flat = uncertainties.flatten()
        z_score = 1.96  # 95%置信区间
        lower_bound = pred_flat - z_score * uncert_flat
        upper_bound = pred_flat + z_score * uncert_flat
        axes[0, 0].fill_between(x_axis, lower_bound[:n_points], upper_bound[:n_points], 
                               alpha=0.2, color='red', label='95% 置信区间')
    
    axes[0, 0].set_title('Time Series Comparison (First 1000 Points)')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Battery Voltage (V)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    n_sample = min(5000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_sample, replace=False)
    sampled_pred = pred_flat[indices]
    sampled_target = target_flat[indices]
    axes[0, 1].scatter(sampled_target, sampled_pred, alpha=0.5, s=1)
    min_val = min(sampled_target.min(), sampled_pred.min())
    max_val = max(sampled_target.max(), sampled_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0, 1].set_title('Predictions vs True Values Scatter Plot')
    axes[0, 1].set_xlabel('True Values (V)')
    axes[0, 1].set_ylabel('Predictions (V)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    errors = pred_flat - target_flat
    axes[1, 0].hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(errors.mean(), color='red', linestyle='--', 
                    label=f'Mean Error: {errors.mean():.4f}')
    axes[1, 0].set_title('Prediction Error Distribution')
    axes[1, 0].set_xlabel('Error (V)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    metric_text = f"""
    Evaluation Metrics:
    Mean Absolute Error (MAE): {metrics['MAE']:.4f} V
    Mean Squared Error (MSE): {metrics['MSE']:.6f} V²
    Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} V
    R-squared (R^2): {metrics['R^2']:.4f}
    Data Statistics:
    Sample Count: {len(pred_flat):,}
    True Value Range: [{target_flat.min():.3f}, {target_flat.max():.3f}] V
    Prediction Range: [{pred_flat.min():.3f}, {pred_flat.max():.3f}] V
    Error Range: [{errors.min():.4f}, {errors.max():.4f}] V
    Model Configuration:
    Sequence Length: {config['data']['sequence_length']}
    Prediction Steps: {config['data']['prediction_steps']}
    Prediction Targets: {config['model']['prediction_targets']}
    Use Uncertainty: {config['model'].get('use_uncertainty', False)}
    Uncertainty Method: {config['model'].get('uncertainty_method', 'N/A')}
    """
    
    # 添加不确定性统计信息
    if uncertainties is not None:
        uncert_flat = uncertainties.flatten()
        # 计算置信区间覆盖率
        z_score = 1.96
        lower_bound = pred_flat - z_score * uncert_flat
        upper_bound = pred_flat + z_score * uncert_flat
        in_interval = (target_flat >= lower_bound) & (target_flat <= upper_bound)
        coverage_rate = np.mean(in_interval)
        
        uncertainty_text = f"""
    Uncertainty Statistics:
    Mean Uncertainty: {uncert_flat.mean():.4f}
    Uncertainty Std: {uncert_flat.std():.4f}
    95% Coverage Rate: {coverage_rate*100:.2f}%
    Uncertainty Range: [{uncert_flat.min():.4f}, {uncert_flat.max():.4f}]
        """
        metric_text += uncertainty_text
    
    axes[1, 1].text(0.1, 0.9, metric_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    return metrics

def plot_detailed_comparison(predictions, targets, uncertainties=None, save_path='detailed_comparison.png'):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Detailed Prediction Analysis', fontsize=16, fontweight='bold')
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    n_detail = min(500, len(pred_flat))
    x_detail = np.arange(n_detail)
    axes[0].plot(x_detail, target_flat[:n_detail], 'b-', label='True Values', linewidth=2)
    axes[0].plot(x_detail, pred_flat[:n_detail], 'r--', label='Predictions', linewidth=2)
    axes[0].fill_between(x_detail, target_flat[:n_detail], pred_flat[:n_detail], 
                        alpha=0.3, color='yellow', label='Error Area')
    
    # 添加置信区间
    if uncertainties is not None:
        uncert_flat = uncertainties.flatten()
        z_score = 1.96  # 95%置信区间
        lower_bound = pred_flat - z_score * uncert_flat
        upper_bound = pred_flat + z_score * uncert_flat
        axes[0].fill_between(x_detail, lower_bound[:n_detail], upper_bound[:n_detail], 
                           alpha=0.2, color='red', label='95% 置信区间')
    
    axes[0].set_title('Local Detailed Comparison (First 500 Points)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Battery Voltage (V)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    window = 50
    if len(pred_flat) > window:
        pred_ma = pd.Series(pred_flat).rolling(window=window).mean()
        target_ma = pd.Series(target_flat).rolling(window=window).mean()
        x_ma = np.arange(len(pred_ma))
        axes[1].plot(x_ma, target_ma, 'b-', label=f'True Values ({window}-point Moving Average)', linewidth=2)
        axes[1].plot(x_ma, pred_ma, 'r-', label=f'Predictions ({window}-point Moving Average)', linewidth=2)
        axes[1].set_title(f'{window}-Point Moving Average Comparison')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Battery Voltage (V)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    errors = pred_flat - target_flat
    cumulative_errors = np.cumsum(np.abs(errors))
    
    axes[2].plot(cumulative_errors, 'g-', linewidth=2)
    axes[2].set_title('Cumulative Absolute Error')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Cumulative Absolute Error')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison chart saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Battery Voltage Prediction Visualization')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                    help='Config file path')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', 
                    help='Model file path')
    parser.add_argument('--data', type=str, default='Datasets/test', 
                    help='Test data path')
    parser.add_argument('--device', type=str, default='auto', 
                    help='Device selection (cpu/cuda/auto)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                    help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    config = load_config(args.config)
    
    model, checkpoint = load_model(args.model, config)
    
    test_dataset = load_test_data(args.data, config)
    
    # 检查是否启用不确定性
    use_uncertainty = config['model'].get('use_uncertainty', False)
    predictions, targets, uncertainties = generate_predictions(
        model, test_dataset, device, use_uncertainty=use_uncertainty
    )
    
    metrics = plot_predictions(
        predictions, targets, config, 
        uncertainties=uncertainties if use_uncertainty else None,
        save_path=output_dir / 'prediction_comparison.png'
    )
    
    plot_detailed_comparison(
        predictions, targets,
        uncertainties=uncertainties if use_uncertainty else None,
        save_path=output_dir / 'detailed_comparison.png'
    )
    
    results = {
        'metrics': metrics,
        'model_info': {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_loss': checkpoint.get('train_loss', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A')
        },
        'data_info': {
            'num_samples': len(test_dataset),
            'sequence_length': config['data']['sequence_length'],
            'prediction_steps': config['data']['prediction_steps']
        }
    }
    
    with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("Visualization completed!")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Main comparison chart: {output_dir}/prediction_comparison.png")
    print(f"Detailed analysis chart: {output_dir}/detailed_comparison.png")
    print(f"Evaluation results: {output_dir}/evaluation_results.json")
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
if __name__ == '__main__':
    main() 