# 引力模型更新说明 (2026-01-06)

## 🎯 修改内容

### 1. **样本过滤** - 排除原地停留数据
```python
# 在 find_gravity_coef.py 的 _create_samples() 中
if current_location != next_location:
    samples.append({'obs': obs, 'target': next_location})
else:
    filtered_count += 1  # 过滤掉原地停留的样本
```

**原因**: 原地停留的样本不提供有用的移动权重信息

---

### 2. **分数计算特殊处理** - 避免"爆炸"
```python
# 在 Gravity.py 和 find_gravity_coef.py 的 _calculate_gravity_score() 中
if current_grid_id == target_grid_id:
    return weight * 10.0  # 固定适中分数
else:
    score = weight * poi_count / distance^2  # 正常公式
```

**原因**: 
- distance 接近0时，`1/distance^2` 会"爆炸"至极大值
- 使用固定值 `weight × 10.0` 表示"留在原地的惯性"

---

### 3. **保存路径变更**
```bash
# 旧路径
/workspace/China_Journal/config/gravity_weights_beijing.json

# 新路径
/workspace/China_Journal/util/gravity_weight/gravity_weights_beijing.json
```

**原因**: 工具生成的文件放在 util 目录更合理

---

## 📝 使用方法

### 拟合权重
```bash
bash scripts/fit_gravity_weights.sh --city beijing --max_samples 1000
```

### 配置使用
在 `config/config.yaml` 中:
```yaml
model:
  gravity:
    weight_config_path: "/workspace/China_Journal/util/gravity_weight/gravity_weights_beijing.json"
```

---

## ✅ 测试验证

运行测试:
```bash
python test_gravity_fitting_updates.py
```

预期输出:
```
✓ All staying-in-place samples successfully filtered
✓ Same location score is fixed at 10.0 (weight * 10.0)
✓ Directory exists: /workspace/China_Journal/util/gravity_weight
ALL TESTS PASSED!
```

---

## 📊 影响

### 样本数量变化
```
原始训练记录: 41984
过滤前样本: ~30000
过滤后样本: ~18000 (过滤掉约40%原地停留样本)
```

### 分数计算
```python
# 原地停留
Grid 516 -> Grid 516: score = 10.0 (固定)

# 正常移动  
Grid 516 -> Grid 517: score = weight × poi_count / distance^2
                             = 1.0 × 60 / 0.4^2
                             = 375.0 → 归一化后 ≈ 0.8
```

---

## 🔄 已修改文件

1. ✅ `/workspace/China_Journal/util/find_gravity_coef.py`
   - `_create_samples()` - 过滤原地停留
   - `_calculate_gravity_score()` - 特殊处理
   - `main()` - 更新默认保存路径

2. ✅ `/workspace/China_Journal/model/Gravity.py`
   - `_calculate_gravity_score()` - 特殊处理原地停留

3. ✅ `/workspace/China_Journal/scripts/fit_gravity_weights.sh`
   - 更新 `OUTPUT_DIR` 默认值

4. ✅ `/workspace/China_Journal/config/config.yaml`
   - 更新 `weight_config_path` 示例路径

5. ✅ `/workspace/China_Journal/docs/GRAVITY_MODEL_UPDATES.md`
   - 更新文档说明

---

## 💡 技术细节

### 为什么过滤原地停留？
1. **无移动信息**: 停留不涉及位置选择
2. **干扰优化**: 大量停留样本会偏向"不移动"的权重
3. **更准确**: 专注于实际移动行为的权重拟合

### 为什么用固定分数10.0？
1. **避免爆炸**: `distance ≈ 0` 时 `1/0.01^2 = 10000`
2. **适中值**: 10.0 既不太高也不太低
3. **可调节**: 通过 weight 仍可调节不同类别的停留倾向

### 数学推导
```
正常情况: score = w × poi / d^2
d → 0:    score → ∞  (爆炸)

改进后:
d = 0:    score = w × 10  (固定)
d > 0:    score = w × poi / d^2  (正常)
```

---

## 🚀 下一步

1. 为三个城市拟合权重:
   ```bash
   bash scripts/fit_gravity_weights.sh --city beijing --max_samples 5000
   bash scripts/fit_gravity_weights.sh --city nanchang --max_samples 5000
   bash scripts/fit_gravity_weights.sh --city shenzhen --max_samples 5000
   ```

2. 在训练中启用:
   ```yaml
   weight_config_path: "util/gravity_weight/gravity_weights_beijing.json"
   ```

3. 对比性能:
   - 固定权重 vs 拟合权重
   - Top-K准确率提升
