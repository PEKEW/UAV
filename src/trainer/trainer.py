import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import numpy as np
from src.models.model import LSTM
# from src.utils.early_stopping import EarlyStopping
import time
from datetime import timedelta
import psutil
import GPUtil
import os
from pathlib import Path




class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self._setup_training()
        self.start_time = time.time()
        self.best_val_loss = float('inf')
        self.epoch_times = []
        
    def _setup_training(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['train']['learning_rate'],
            weight_decay=self.config['train']['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config['train']['scheduler_mode'],
            factor=self.config['train']['scheduler_factor'],
            patience=self.config['train']['scheduler_patience'],
            verbose=self.config['train']['scheduler_verbose']
        )
        
        self.device = torch.device(self.config['train']['device'])
        self.model.to(self.device)
        
    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _compute_loss")
    
    def _get_gpu_info(self):
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                'memory_used': f"{gpu.memoryUsed}MB",
                'memory_total': f"{gpu.memoryTotal}MB",
                'gpu_load': f"{gpu.load*100:.1f}%",
                'temperature': f"{gpu.temperature}°C"
            }
        except:
            return {'error': '无法获取GPU信息'}
    
    def _get_system_info(self):
        return {
            'cpu_percent': f"{psutil.cpu_percent()}%",
            'memory_percent': f"{psutil.virtual_memory().percent}%"
        }
    
    def _format_time(self, seconds):
        return str(timedelta(seconds=int(seconds)))
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        batch_times = []
        
        pbar = tqdm(dataloader, 
                    desc=f'Epoch {epoch+1}/{self.config["train"]["num_epochs"]}',
                    leave=True,  
                    position=0,  
                    ncols=150,  
                    mininterval=0.1)  
        
        for batch_idx, (x, y) in enumerate(pbar):
            batch_start_time = time.time()
            
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self._compute_loss(outputs, y)
            loss.backward()
            
            if 'grad_clip' in self.config['train']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['train']['grad_clip']
                )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            # 计算批次时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更新进度条信息
            if batch_idx % 5 == 0:  # 每5个批次更新一次系统信息
                gpu_info = self._get_gpu_info()
                sys_info = self._get_system_info()
                avg_batch_time = np.mean(batch_times[-5:])
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}',
                    'batch_time': f'{avg_batch_time:.3f}s',
                    'GPU_mem': gpu_info.get('memory_used', 'N/A'),
                    'GPU_load': gpu_info.get('gpu_load', 'N/A'),
                    'CPU': sys_info['cpu_percent'],
                    'RAM': sys_info['memory_percent']
                }, refresh=True)  
                
                pbar.refresh()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc='Validation'):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self._compute_loss(outputs, y)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Tuple[List[float], List[float]]:
        train_losses = []
        val_losses = []
        
        print("\n开始训练...")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.config['train']['batch_size']}")
        print(f"学习率: {self.config['train']['learning_rate']}")
        print(f"总批次数: {len(train_dataloader)}")
        print(f"总epoch数: {self.config['train']['num_epochs']}")
        print("-" * 50)
        
        for epoch in range(self.config['train']['num_epochs']):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(train_dataloader, epoch)
            train_losses.append(train_loss)
            
            val_loss = self.validate(val_dataloader)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = self.config['train']['num_epochs'] - (epoch + 1)
            estimated_time = avg_epoch_time * remaining_epochs
            
            print(f"\nEpoch {epoch+1}/{self.config['train']['num_epochs']} 总结:")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Epoch用时: {self._format_time(epoch_time)}")
            print(f"预计剩余时间: {self._format_time(estimated_time)}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, train_loss, val_loss)
                print(f"保存最佳模型 (验证损失: {val_loss:.4f})")
            
            print("-" * 50)
        
        total_time = time.time() - self.start_time
        print("\n训练完成!")
        print(f"总训练时间: {self._format_time(total_time)}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        save_path = Path(self.config['train']['model_save_path'])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
    
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                outputs = self.model(x)
                self._process_predictions(outputs, predictions)
        
        for key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)
            
        return predictions
    
    def _process_predictions(self, outputs: Any, predictions: Dict[str, List[np.ndarray]]):
        raise NotImplementedError("Subclasses must implement _process_predictions")



class LSTMTrainer(Trainer):
    def __init__(self, model: LSTM, config: Dict[str, Any]):
        super().__init__(model, config)
        self.criterion = nn.MSELoss()
    
    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if self.model.use_multi_task:
            loss = 0
            for target_name in self.model.prediction_targets:
                target_idx = self.model.feature_columns.index(target_name)
                target_feature = targets[:, :, target_idx:target_idx+1].squeeze(-1)  # [batch_size, prediction_steps]
                loss += self.criterion(outputs[target_name], target_feature)
            return loss
        else:
            if outputs.shape != targets.shape:
                target_idx = self.model.feature_columns.index(self.model.prediction_targets[0])
                targets = targets[:, :, target_idx:target_idx+1].squeeze(-1)  # [batch_size, prediction_steps]
            return self.criterion(outputs, targets)
    
    def _process_predictions(self, outputs: Any, predictions: Dict[str, List[np.ndarray]]):
        if self.model.use_multi_task:
            for target in self.model.prediction_targets:
                if target not in predictions:
                    predictions[target] = []
                predictions[target].append(outputs[target].cpu().numpy())
        else:
            if 'predictions' not in predictions:
                predictions['predictions'] = []
            predictions['predictions'].append(outputs.cpu().numpy())