"""
损失函数模块
包含异常检测任务的专用损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        p_t = torch.exp(-ce_loss)
        
        # 计算alpha权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        # 计算focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        batch_size, num_classes = inputs.shape
        
        # 计算log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 创建smooth targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算损失
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for class imbalance
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        if self.class_weights is not None:
            if self.class_weights.device != inputs.device:
                self.class_weights = self.class_weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary classification
    """
    
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        # 转换为概率
        probs = F.softmax(inputs, dim=1)
        
        # 转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        # 计算Dice系数
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回Dice损失
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        # 转换为概率
        probs = F.softmax(inputs, dim=1)
        
        # 转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        # 计算True Positive, False Positive, False Negative
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)
        
        # 计算Tversky系数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function
    """
    
    def __init__(self, losses: list, weights: list):
        super().__init__()
        assert len(losses) == len(weights), "损失函数数量必须与权重数量相等"
        
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        total_loss = 0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    可以用于不平衡数据的异常检测
    """
    
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出logits
            targets: [batch_size] - 真实标签
        """
        # 转换为概率
        probs = torch.sigmoid(inputs)
        
        # 转换targets为one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        # 计算正负样本的损失
        xs_pos = probs
        xs_neg = 1 - probs
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # 计算损失
        los_pos = targets_one_hot * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets_one_hot) * torch.log(xs_neg.clamp(min=1e-8))
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets_one_hot
            pt1 = xs_neg * (1 - targets_one_hot)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets_one_hot + self.gamma_neg * (1 - targets_one_hot)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            los_pos *= one_sided_w
            los_neg *= one_sided_w
        
        loss = los_pos + los_neg
        return -loss.sum(dim=1).mean()