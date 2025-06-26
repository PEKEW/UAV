# CNN-LSTM时序分类模型完整架构方案

## 模型整体流程
输入：30秒时序数据 [batch_size, sequence_length, features]  
输出：类别概率分布 [batch_size, num_classes]

## 1. 输入预处理层
- **输入规范化**：对每个特征维度进行标准化（零均值，单位方差）
- **滑动窗口分割**：
  - 窗口大小：1秒
  - 步长：0.5秒（50%重叠）
  - 生成59个窗口，每个窗口包含原始特征

## 2. 多尺度CNN特征提取
**并行卷积分支**：
- 分支1：Conv1D(features→128, kernel_size=3, padding='same')
- 分支2：Conv1D(features→128, kernel_size=5, padding='same')  
- 分支3：Conv1D(features→128, kernel_size=7, padding='same')
- 每个分支后：BatchNorm → ReLU → Dropout(0.2)
- **输出融合**：拼接3个分支 → Conv1D(384→256, kernel_size=1)

## 3. 深度卷积网络
**第1个残差块**：
- Conv1D(256→256, kernel_size=3, padding='same')
- BatchNorm → ReLU → Dropout(0.2)
- Conv1D(256→256, kernel_size=3, padding='same')
- BatchNorm → 残差相加 → ReLU
- MaxPool1D(pool_size=2, stride=2)

**第2个残差块**：
- Conv1D(256→512, kernel_size=3, padding='same')
- BatchNorm → ReLU → Dropout(0.2)
- Conv1D(512→512, kernel_size=3, padding='same')
- BatchNorm → ReLU
- 残差路径：Conv1D(256→512, kernel_size=1) → 相加
- MaxPool1D(pool_size=2, stride=2)

**第3个残差块**：
- Conv1D(512→512, kernel_size=3, padding='same')
- BatchNorm → ReLU → Dropout(0.3)
- Conv1D(512→512, kernel_size=3, padding='same')
- BatchNorm → 残差相加 → ReLU
- MaxPool1D(pool_size=2, stride=2)

**输出形状**：[batch_size, 512, sequence_length//8]

## 4. 特征重组层
- Conv1D(512→256, kernel_size=1)：降低特征维度
- Transpose：[batch_size, sequence_length//8, 256]
- 序列长度：约7-8个时间步（取决于原始序列长度）

## 5. 双向LSTM序列建模
**第1层双向LSTM**：
- 前向LSTM：256个隐藏单元
- 后向LSTM：256个隐藏单元
- 输出：拼接前后向，得到512维
- LayerNorm → Dropout(0.4)

**第2层双向LSTM**：
- 前向LSTM：256个隐藏单元
- 后向LSTM：256个隐藏单元
- 输出：拼接前后向，得到512维
- LayerNorm → Dropout(0.4)
- 返回所有时间步的输出

## 6. 注意力池化层
- **注意力分数计算**：
  - Linear(512→256) → Tanh → Linear(256→1)
  - 对时间维度应用Softmax得到注意力权重
- **加权聚合**：
  - 使用注意力权重对所有时间步的LSTM输出加权求和
  - 输出：[batch_size, 512]

## 7. 分类头
**第1层全连接**：
- Linear(512→256)
- BatchNorm → LeakyReLU(0.01) → Dropout(0.5)

**第2层全连接**：
- Linear(256→128)
- BatchNorm → LeakyReLU(0.01) → Dropout(0.5)

**输出层**：
- Linear(128→num_classes)
- 训练时：返回logits
- 推理时：Softmax激活

## 8. 训练配置

**损失函数**：
- CrossEntropyLoss（包含Softmax）
- 标签平滑系数：0.1

**优化器**：
- AdamW优化器
- 初始学习率：1e-3
- 权重衰减：1e-4

**学习率调度**：
- 前5个epoch：线性warmup
- 之后：余弦退火，最小学习率1e-5

**梯度裁剪**：
- 梯度范数裁剪：5.0

**批量大小**：
- 训练：32
- 推理：64

**早停策略**：
- 监控验证集损失
- 耐心值：10个epoch

## 9. 数据增强（训练时）
- 时间拉伸：0.9-1.1倍
- 高斯噪声：std=0.01
- 随机遮挡：10%的时间步置零