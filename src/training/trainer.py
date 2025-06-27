import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from collections import defaultdict
import wandb

from .loss_functions import FocalLoss, LabelSmoothingLoss
from .metrics import AnomalyMetrics, MetricsTracker
from .model_utils import check_model_health, reset_corrupted_weights, safe_model_forward
from .scheduler import WarmupCosineScheduler


class BaseTrainer:
    """训练器基类"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.epochs = int(config.get('epochs', 100))
        self.learning_rate = float(config.get('learning_rate', 1e-3))
        self.weight_decay = float(config.get('weight_decay', 1e-4))
        self.gradient_clip = float(config.get('gradient_clip', 2.0))
        self.accumulation_steps = int(config.get('accumulation_steps', 4))
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 早停配置
        self.early_stopping = config.get('early_stopping', True)
        self.patience = config.get('patience', 15)
        self.monitor_metric = config.get('monitor_metric', 'val_f1')
        self.monitor_mode = config.get('monitor_mode', 'max')
        
        # 保存配置
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志配置
        self.use_wandb = config.get('use_wandb', False)
        self.use_tensorboard = config.get('use_tensorboard', True)  # 默认启用TensorBoard
        self.log_interval = config.get('log_interval', 100)
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 初始化组件
        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_metrics()
        self._setup_logging()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf') if self.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
    def _setup_loss_function(self):
        """设置损失函数"""
        loss_type = self.config.get('loss_function', 'focal')
        # Test: 固定损失函数
        if False and loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 0.25),
                gamma=self.config.get('focal_gamma', 2.0)
            )
        elif False and loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(
                smoothing=self.config.get('label_smoothing', 0.1)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.criterion.to(self.device)
        
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        # Test: 固定优化器
        if True or optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.config.get('adamw_betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        # Test: 固定学习率调度器
        if True or scheduler_type == 'cosine':
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 10),
                max_epochs=self.epochs,
                eta_min=self.config.get('min_lr', 1e-3)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.7)
            )
        else:
            self.scheduler = None
    
    def _setup_metrics(self):
        """设置评估指标"""
        self.metrics = AnomalyMetrics()
        self.metrics_tracker = MetricsTracker()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化TensorBoard
        if self.use_tensorboard:
            tensorboard_dir = self.save_dir.parent / 'tensorboard_logs'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            self.logger.info(f"TensorBoard日志目录: {tensorboard_dir}")
        else:
            self.writer = None
        
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'battery-anomaly-detection'),
                config=self.config
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        total_norm = 0
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                if torch.isnan(param_norm) or torch.isinf(param_norm):
                    self.logger.warning(f"参数 {name} 包含NaN或Inf")
        
        total_norm = total_norm ** (1. / 2)
        if param_count > 0:
            self.logger.debug(f"训练前模型权重范数: {total_norm:.6f}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            batch_metrics = self.metrics.compute_batch_metrics(output, target)
            for key, value in batch_metrics.items():
                if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                    self.logger.warning(f"发现NaN或Inf指标值 {key}: {value}")
                    value = 0.0
                epoch_metrics[key] += value
            
            loss_value = loss.item()
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                self.logger.warning(f"发现NaN或Inf损失值: {loss_value}")
                loss_value = 0.0
            
            epoch_metrics['loss'] += loss_value
            
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}'
                )
            
            self.global_step += 1
        
        for key in epoch_metrics:
            if num_batches > 0:
                if torch.isnan(torch.tensor(epoch_metrics[key])):
                    self.logger.warning(f"指标 {key} 在累积时已经为NaN: {epoch_metrics[key]}")
                epoch_metrics[key] /= num_batches
                if torch.isnan(torch.tensor(epoch_metrics[key])):
                    self.logger.warning(f"指标 {key} 在平均后变为NaN: {epoch_metrics[key]}")
            else:
                self.logger.warning(f"num_batches为0，无法计算平均指标")
                epoch_metrics[key] = float('nan')
        
        return dict(epoch_metrics)
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # output = safe_model_forward(self.model, data)
                output = self.model(data)
                
                if output is None:
                    self.logger.warning(f"验证批次 {batch_idx}: 安全前向传播失败")
                    if not hasattr(self, '_nan_batch_count'):
                        self._nan_batch_count = 0
                    self._nan_batch_count += 1
                    continue
                
                loss = self.criterion(output, target)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    self.logger.warning(f"验证批次 {batch_idx}: 损失为NaN或Inf, 输出范围: [{output.min():.6f}, {output.max():.6f}]")
                    continue
                
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
                
                epoch_metrics['loss'] += loss.item()
        
        if len(all_outputs) == 0:
            self.logger.error("所有验证批次都包含NaN，无法计算验证指标")
            return {'loss': float('inf'), 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
        if hasattr(self, '_nan_batch_count') and self._nan_batch_count > 0:
            self.logger.warning(f"验证中跳过了 {self._nan_batch_count} 个包含NaN的批次")
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        overall_metrics = self.metrics.compute_epoch_metrics(all_outputs, all_targets)
        
        for key, value in overall_metrics.items():
            if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                self.logger.warning(f"验证中发现NaN或Inf指标值 {key}: {value}")
                value = 0.0
            epoch_metrics[key] = value
        
        if num_batches > 0:
            epoch_metrics['loss'] /= num_batches
        else:
            epoch_metrics['loss'] = float('nan')
        
        return dict(epoch_metrics)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.logger.info(f"开始训练 - 设备: {self.device}")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = val_metrics.get(self.monitor_metric.replace('val_', ''), 
                                                train_metrics.get(self.monitor_metric.replace('train_', ''), 0))
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Test: 不用早停
            if False and self._check_early_stopping(val_metrics):
                self.logger.info(f"早停触发，在第{epoch}轮停止训练")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        
        if self.writer is not None:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float], epoch_time: float):
        train_loss = train_metrics.get('loss', 0)
        val_loss = val_metrics.get('loss', 0)
        train_f1 = train_metrics.get('f1', 0)
        val_f1 = val_metrics.get('f1', 0)
        train_acc = train_metrics.get('accuracy', 0)
        val_acc = val_metrics.get('accuracy', 0)
        
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.training_history[f'val_{key}'].append(value)
        
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            self.writer.add_scalar('Time/Epoch_Time', epoch_time, epoch)
            
            if epoch % 10 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'epoch_time': epoch_time,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            for key, value in train_metrics.items():
                log_dict[f'train_{key}'] = value
            for key, value in val_metrics.items():
                log_dict[f'val_{key}'] = value
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'training_history': dict(self.training_history)
        }
        
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        current_metric = val_metrics.get(self.monitor_metric.replace('val_', ''), 
                                    train_metrics.get(self.monitor_metric.replace('train_', ''), 0))
        
        is_best = False
        if self.monitor_mode == 'max':
            is_best = current_metric > self.best_metric
        else:
            is_best = current_metric < self.best_metric
            
        if is_best:
            self.best_metric = current_metric
            torch.save(checkpoint, self.save_dir / 'best.pth')
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否早停"""
        if not self.early_stopping:
            return False
        
        return self.patience_counter >= self.patience
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
        
        self.logger.info(f"已加载检查点: {checkpoint_path}, Epoch: {checkpoint['epoch']}")


class BatteryTrainer(BaseTrainer):
    """电池模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        battery_config = {
            'loss_function': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'learning_rate': 1e-3,
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'monitor_metric': 'val_f1',
            'patience': 15
        }
        battery_config.update(config)
        
        super().__init__(model, battery_config)


class FlightTrainer(BaseTrainer):
    """飞行模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        # 飞行特定配置
        flight_config = {
            'loss_function': 'label_smoothing',
            'label_smoothing': 0.1,
            'learning_rate': 8e-4,
            'scheduler': 'step',
            'step_size': 10,
            'gamma': 0.7,
            'monitor_metric': 'val_auc',
            'patience': 12
        }
        flight_config.update(config)
        
        super().__init__(model, flight_config)


class DualDomainTrainer:
    """双域训练器"""
    
    def __init__(self, battery_model: nn.Module, flight_model: nn.Module, config: Dict[str, Any]):
        self.battery_model = battery_model
        self.flight_model = flight_model
        self.config = config
        
        self.battery_trainer = BatteryTrainer(battery_model, config.get('battery_config', {}))
        self.flight_trainer = FlightTrainer(flight_model, config.get('flight_config', {}))

        # 并发训练配置
        self.concurrent_training = config.get('concurrent_training', False)
        
    def train(self, battery_loaders: Tuple[DataLoader, DataLoader], 
              flight_loaders: Tuple[DataLoader, DataLoader]):
        """训练两个模型"""
        battery_train_loader, battery_val_loader = battery_loaders
        flight_train_loader, flight_val_loader = flight_loaders
        
        if self.concurrent_training:
            # TODO: 实现并发训练
            raise NotImplementedError("并发训练功能待实现")
        else:
            # 顺序训练
            print("开始训练电池异常检测模型...")
            # Test 电池分类模型已经训练好了 不需要再训练
            # self.battery_trainer.train(battery_train_loader, battery_val_loader)
            
            print("开始训练飞行异常检测模型...")
            self.flight_trainer.train(flight_train_loader, flight_val_loader)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        return {
            'battery_training': {
                'best_metric': self.battery_trainer.best_metric,
                'history': dict(self.battery_trainer.training_history)
            },
            'flight_training': {
                'best_metric': self.flight_trainer.best_metric,
                'history': dict(self.flight_trainer.training_history)
            }
        }