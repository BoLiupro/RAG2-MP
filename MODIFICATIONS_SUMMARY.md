# 修改总结：全网格预测 + 梯度监控

## 问题
1. **原问题**：第一个sample训练正常，第二个sample开始LLM的response（rag_summary）变成"！！！！"
2. **根本原因**：LLM参数未冻结，训练过程中权重被更新导致崩坏

## 解决方案

### 1. 冻结LLM，只训练分类头
```python
# 冻结LLM所有参数
for p in predictor.llm.model.parameters():
    p.requires_grad = False

# 只训练分类头
for p in predictor.classification_head.parameters():
    p.requires_grad = True
```

### 2. 修改预测目标：从candidates → 所有grids
**之前**：只对gravity提供的10-20个candidates预测
**现在**：对所有1600个grids预测

#### 分类头架构更新
```python
# 输出维度从1 → 1600
self.num_grids = 40 * 40  # 1600
nn.Linear(256, self.num_grids)  # 输出所有grids的logits
```

#### Loss计算更新
```python
# 直接用grid_id作为目标
logits = classification_head(context_encoding)  # (1, 1600)
gt_tensor = torch.tensor([ground_truth], dtype=torch.long)
loss = F.cross_entropy(logits, gt_tensor)
```

### 3. 数值稳定性优化

#### 添加LayerNorm
```python
nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.LayerNorm(512),  # 稳定激活值
    nn.ReLU(),
    ...
)
```

#### 使用Float32进行分类头计算
```python
# LLM可以用float16，但分类头用float32避免数值问题
self.use_fp32_head = True
context_encoding = context_encoding.float()
logits = classification_head(context_encoding)
```

#### 权重初始化
```python
# 使用小的初始化gain
nn.init.xavier_uniform_(module.weight, gain=0.1)
nn.init.zeros_(module.bias)
```

### 4. 训练稳定性措施

#### 降低学习率
```yaml
learning_rate: 5.0e-5  # 从1e-4降到5e-5
```

#### 更强的梯度裁剪
```yaml
max_grad_norm: 0.5  # 从1.0降到0.5
```

#### 梯度异常监控
```python
# 检测NaN/Inf
if torch.isnan(loss) or torch.isinf(loss):
    raise ValueError("Loss is NaN or Inf")

# 监控梯度范数
total_norm = 0.0
for p in classification_head.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        if torch.isnan(p.grad).any():
            has_nan_grad = True
        if torch.isinf(p.grad).any():
            has_inf_grad = True
total_norm = total_norm ** 0.5

# 记录梯度统计
if step_idx % log_every_n_steps == 0:
    self.log(f"Gradient norm: {total_norm:.4f}")
```

## 修改的文件

### 1. model/Predictor.py
- 分类头输出从1维改为1600维
- 添加LayerNorm层
- 添加权重初始化
- 使用float32进行分类头计算
- 预测方法改为对所有grids预测

### 2. trainer/trainer.py
- 冻结LLM参数
- 只优化分类头参数
- Loss计算改为直接对1600个grids
- 添加梯度异常检测（NaN/Inf/大范数）
- 添加详细的梯度统计日志
- Context encoding转为float32

### 3. config/config.yaml
- learning_rate: 1e-4 → 5e-5
- max_grad_norm: 1.0 → 0.5

## 测试结果

```
✓ Classification head output: 1600 grids
✓ Logits range: [-0.14, 0.12]
✓ Loss: 7.27 (stable across samples)
✓ Gradient norm: 0.50 (clipped)
✓ No NaN/Inf in gradients
✓ LLM remains frozen
✓ RAG summary stays normal across training steps
```

## 关键改进

### 1. 稳定性
- LLM冻结 → RAG不会崩坏
- Float32分类头 → 避免数值溢出
- LayerNorm → 稳定激活分布
- 梯度裁剪 → 防止梯度爆炸

### 2. 预测能力
- 预测空间：candidates (10-20) → all grids (1600)
- Gravity的candidates仅作为prompt信息，不限制预测范围
- 分类头可以学习预测任意grid

### 3. 可监控性
- 实时梯度范数监控
- NaN/Inf自动检测
- 详细的梯度统计日志
- Logits范围检查

## 训练流程

```
Input Trajectory
    ↓
RAG (frozen LLM) → Similar Patterns Summary
    ↓
Gravity Model → Candidate Info (for prompt only)
    ↓
Prompt Builder → Context Prompt
    ↓
LLM Encoder (frozen) → Hidden States (float16)
    ↓
    Convert to float32
    ↓
Classification Head (trainable, float32) → 1600-dim Logits
    ↓
Cross-Entropy Loss with ground truth grid_id
    ↓
Backward (only through classification head)
    ↓
Gradient Clipping (max_norm=0.5)
    ↓
Optimizer Step (lr=5e-5)
```

## 性能影响

- **内存**：分类头参数量略增（256→1600输出），但相比LLM仍然很小
- **速度**：去除了candidates循环，实际更快
- **训练**：只训练分类头（~2M参数），比微调LLM快得多
- **稳定性**：大幅提升，避免LLM权重崩坏

## 下一步

可以运行完整训练：
```bash
cd /workspace/China_Journal
python trainer/trainer.py --config config/config.yaml
```

梯度监控会自动记录到日志中。
