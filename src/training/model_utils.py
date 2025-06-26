"""
模型工具函数
包含模型诊断和修复功能
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def check_model_health(model: nn.Module) -> Dict[str, Any]:
    """检查模型健康状态"""
    health_info = {
        'has_nan_weights': False,
        'has_inf_weights': False,
        'weight_norm': 0.0,
        'grad_norm': 0.0,
        'corrupted_params': []
    }
    
    total_norm = 0
    grad_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        # 检查权重
        if torch.isnan(param.data).any():
            health_info['has_nan_weights'] = True
            health_info['corrupted_params'].append(f"{name}:weights")
            
        if torch.isinf(param.data).any():
            health_info['has_inf_weights'] = True
            health_info['corrupted_params'].append(f"{name}:weights")
        
        # 计算权重范数
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
        param_count += 1
        
        # 检查梯度
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health_info['corrupted_params'].append(f"{name}:gradients")
            
            if torch.isinf(param.grad).any():
                health_info['corrupted_params'].append(f"{name}:gradients")
            
            grad_norm += param.grad.data.norm(2).item() ** 2
    
    health_info['weight_norm'] = total_norm ** 0.5 if param_count > 0 else 0
    health_info['grad_norm'] = grad_norm ** 0.5 if param_count > 0 else 0
    
    return health_info


def reset_corrupted_weights(model: nn.Module, std: float = 0.02) -> int:
    """重置损坏的权重"""
    reset_count = 0
    
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            logger.warning(f"重置损坏的参数: {name}")
            # 使用正态分布重新初始化
            with torch.no_grad():
                param.data.normal_(0, std)
            reset_count += 1
    
    return reset_count


def clip_model_weights(model: nn.Module, max_norm: float = 10.0) -> bool:
    """裁剪模型权重防止爆炸"""
    clipped = False
    
    for param in model.parameters():
        param_norm = param.data.norm(2)
        if param_norm > max_norm:
            param.data.mul_(max_norm / param_norm)
            clipped = True
    
    return clipped


def safe_model_forward(model: nn.Module, x: torch.Tensor, 
                      max_attempts: int = 3) -> Optional[torch.Tensor]:
    """安全的模型前向传播，带重试机制"""
    
    for attempt in range(max_attempts):
        try:
            # 检查输入
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning(f"输入包含NaN或Inf，尝试 {attempt + 1}/{max_attempts}")
                return None
            
            # 前向传播
            output = model(x)
            
            # 检查输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(f"输出包含NaN或Inf，尝试 {attempt + 1}/{max_attempts}")
                
                if attempt < max_attempts - 1:
                    # 重置损坏的权重并重试
                    reset_count = reset_corrupted_weights(model)
                    if reset_count > 0:
                        logger.info(f"重置了 {reset_count} 个损坏参数，重试前向传播")
                        continue
                return None
            
            return output
            
        except Exception as e:
            logger.warning(f"前向传播异常，尝试 {attempt + 1}/{max_attempts}: {e}")
            if attempt < max_attempts - 1:
                reset_corrupted_weights(model)
            continue
    
    return None