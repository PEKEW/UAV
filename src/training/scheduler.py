"""
学习率调度器模块
包含各种学习率调度策略
"""

import torch
import torch.optim as optim
import math
from typing import Optional


class WarmupCosineScheduler:
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, max_epochs: int, 
                 eta_min: float = 0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # 初始化
        self.step(0)
    
    def get_lr(self, epoch: int):
        """计算当前epoch的学习率"""
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # 计算新的学习率
        new_lrs = self.get_lr(epoch)
        
        # 应用新的学习率
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)


class WarmupStepScheduler:
    """
    带预热的阶梯学习率调度器
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, 
                 step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # 初始化
        self.step(0)
    
    def get_lr(self, epoch: int):
        """计算当前epoch的学习率"""
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 阶梯衰减阶段
            decay_epochs = epoch - self.warmup_epochs
            decay_factor = self.gamma ** (decay_epochs // self.step_size)
            
            return [base_lr * decay_factor for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # 计算新的学习率
        new_lrs = self.get_lr(epoch)
        
        # 应用新的学习率
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'warmup_epochs': self.warmup_epochs,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)


class PolynomialLRScheduler:
    """
    多项式学习率调度器
    """
    
    def __init__(self, optimizer: optim.Optimizer, max_epochs: int, 
                 power: float = 0.9, eta_min: float = 0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # 初始化
        self.step(0)
    
    def get_lr(self, epoch: int):
        """计算当前epoch的学习率"""
        progress = min(epoch / self.max_epochs, 1.0)
        decay_factor = (1 - progress) ** self.power
        
        return [
            self.eta_min + (base_lr - self.eta_min) * decay_factor
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # 计算新的学习率
        new_lrs = self.get_lr(epoch)
        
        # 应用新的学习率
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'max_epochs': self.max_epochs,
            'power': self.power,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)


class CyclicLRScheduler:
    """
    循环学习率调度器
    """
    
    def __init__(self, optimizer: optim.Optimizer, base_lr: float, max_lr: float,
                 step_size_up: int, step_size_down: Optional[int] = None, 
                 mode: str = 'triangular', gamma: float = 1.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        
        self.total_size = self.step_size_up + self.step_size_down
        self.step_num = 0
        
    def get_lr(self):
        """计算当前的学习率"""
        cycle = math.floor(1 + self.step_num / self.total_size)
        x = abs(self.step_num / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2.0 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.step_num
        else:
            scale_factor = 1.0
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        
        return lr
    
    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'step_size_up': self.step_size_up,
            'step_size_down': self.step_size_down,
            'mode': self.mode,
            'gamma': self.gamma,
            'step_num': self.step_num
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)


class OneCycleLRScheduler:
    """
    One Cycle学习率调度器
    """
    
    def __init__(self, optimizer: optim.Optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos', 
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.step_num = 0
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        
        # 计算阶段步数
        self.step_up_size = int(self.pct_start * total_steps)
        self.step_down_size = total_steps - self.step_up_size
    
    def get_lr(self):
        """计算当前的学习率"""
        if self.step_num <= self.step_up_size:
            # 上升阶段
            progress = self.step_num / self.step_up_size
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # 下降阶段
            progress = (self.step_num - self.step_up_size) / self.step_down_size
            
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) / 2
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.final_lr) * progress
        
        return lr
    
    def step(self):
        """更新学习率"""
        if self.step_num < self.total_steps:
            lr = self.get_lr()
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.step_num += 1
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'max_lr': self.max_lr,
            'total_steps': self.total_steps,
            'pct_start': self.pct_start,
            'anneal_strategy': self.anneal_strategy,
            'div_factor': self.div_factor,
            'final_div_factor': self.final_div_factor,
            'step_num': self.step_num
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)