# Quick Reference Guide - 快速参考指南

## Modified Files - 修改的文件

### Model Files (模型文件)
- ✅ `model/LLM.py` - Removed all print statements
- ✅ `model/RAG.py` - Removed all print statements  
- ✅ `model/Predictor.py` - Removed all print statements
- ✅ `model/Gravity.py` - Removed all print statements

### Trainer Files (训练脚本)
- ✅ `trainer/trainer.py` - Enhanced logging, config compatibility
- ✅ `trainer/detailed_trainer.py` - NEW FILE for verbose debugging

### Configuration (配置文件)
- ✅ `config/config.yaml` - Expanded with comprehensive parameters

---

## Usage Examples - 使用示例

### 1. Normal Training (正常训练)
```bash
cd /workspace/China_Journal
python trainer/trainer.py --config config/config.yaml
```

### 2. Detailed Debugging (详细调试)
```bash
cd /workspace/China_Journal
python trainer/detailed_trainer.py --config config/config.yaml
```

### 3. Test with Small Sample (小样本测试)
Modify config.yaml:
```yaml
data:
  max_test_samples: 10

detailed_debug:
  num_samples: 5
```

Then run:
```bash
python trainer/detailed_trainer.py --config config/config.yaml
```

---

## Config Structure - 配置结构

### Old Format (旧格式) - Still Supported
```yaml
model:
  llm_model_name: "Deepseek-R1-Distill-Qwen-3B"
  llm_model_path: "/datadisk"
  rag_top_m_samples: 5
  gravity_top_n_candidates: 5
  # ...
```

### New Format (新格式) - Recommended
```yaml
model:
  llm:
    model_name: "Deepseek-R1-Distill-Qwen-3B"
    model_path: "/datadisk"
    # ...
  rag:
    top_m_samples: 5
    # ...
  gravity:
    top_n_candidates: 5
    # ...
  prediction:
    top_k_predictions: 10
```

---

## Key Parameters - 关键参数

### Data Parameters (数据参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.city` | City name | "beijing" |
| `data.obs_len` | Observation length | 12 |
| `data.pred_len` | Prediction length | 1 |
| `data.max_train_samples` | Max training samples | 1000 |

### LLM Parameters (LLM参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.llm.model_name` | LLM model name | "Deepseek-R1-Distill-Qwen-3B" |
| `model.llm.max_new_tokens` | Max generation tokens | 512 |
| `model.llm.temperature` | Sampling temperature | 0.7 |
| `model.llm.use_quantization` | Use 4-bit quantization | true |

### RAG Parameters (RAG参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.rag.top_m_samples` | Number of similar samples | 5 |
| `model.rag.database_path` | RAG database path | "/workspace/China_Journal/util/rag_database" |
| `model.rag.max_summary_length` | Max summary length | 500 |

### Gravity Parameters (Gravity参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.gravity.top_n_candidates` | Candidates per category | 5 |
| `model.gravity.weight` | Weight in gravity formula | 1.0 |
| `model.gravity.radius` | Search radius (grids) | 10 |

### Training Parameters (训练参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.num_epochs` | Number of epochs | 10 |
| `training.learning_rate` | Learning rate | 1.0e-4 |
| `training.samples_per_epoch` | Samples per epoch | 500 |
| `training.log_every_n_steps` | Log frequency | 10 |

### Debug Parameters (调试参数)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `detailed_debug.num_samples` | Number of debug samples | 10 |
| `detailed_debug.print_prompts` | Print LLM prompts | true |
| `detailed_debug.print_rag_results` | Print RAG results | true |
| `detailed_debug.print_gravity_candidates` | Print gravity candidates | true |

---

## Output Files - 输出文件

### Training (训练)
- `output/logs/training_YYYYMMDD_HHMMSS.log` - Training log
- `output/logs/metrics.json` - Training metrics
- `output/checkpoints/checkpoint_epoch_N.pt` - Model checkpoints
- `output/checkpoints/best_model.pt` - Best model

### Testing (测试)
- `output/logs/test_detailed_YYYYMMDD_HHMMSS.log` - Test log

### Detailed Debugging (详细调试)
- `output/logs/detailed_debug_YYYYMMDD_HHMMSS.log` - Debug log
- `output/logs/detailed_test_results.json` - Test results JSON

---

## Common Commands - 常用命令

### View Recent Logs (查看最新日志)
```bash
# View training log
tail -f output/logs/training_*.log

# View debug log
tail -f output/logs/detailed_debug_*.log
```

### Check Config Syntax (检查配置语法)
```bash
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

### Count Samples (统计样本数)
```bash
python -c "
from dataset.dataset import load_datasets
train, val, test = load_datasets('beijing', 'data', 12, 1)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
"
```

---

## Troubleshooting - 故障排除

### Problem: Model prints too much information
**Solution:** Model code no longer prints. Use trainer logging instead.

### Problem: Want to see intermediate results
**Solution:** Use `detailed_trainer.py` instead of `trainer.py`

### Problem: Config file error
**Solution:** Check YAML syntax and parameter names. Use new format for better organization.

### Problem: Out of memory
**Solution:** 
1. Set `model.llm.use_quantization: true`
2. Reduce `training.samples_per_epoch`
3. Reduce `model.llm.max_new_tokens`

### Problem: Training too slow
**Solution:**
1. Use smaller `max_train_samples` for testing
2. Enable GPU if available
3. Reduce `training.samples_per_epoch`

---

## Code Changes Summary - 代码变更摘要

### Removed from Models (从模型中移除)
- All `print()` statements
- Verbose logging during initialization
- Intermediate result printing

### Added to Trainers (添加到训练器)
- Enhanced logging in `trainer.py`
- New `detailed_trainer.py` for debugging
- Config format compatibility check
- Detailed progress tracking

### Enhanced in Config (配置增强)
- Hierarchical organization (llm, rag, gravity, prediction)
- More comprehensive parameters
- Debug-specific configuration
- Better documentation

---

## Best Practices - 最佳实践

### For Development (开发阶段)
1. Use `detailed_trainer.py` with small samples
2. Enable all debug printing options
3. Check logs frequently
4. Use `max_test_samples: 10` for quick iteration

### For Production (生产环境)
1. Use `trainer.py` for actual training
2. Set `verbose: false` to reduce output
3. Use full datasets (`max_samples: null`)
4. Save checkpoints regularly

### For Debugging (调试阶段)
1. Start with 1-2 samples
2. Enable `print_prompts: true`
3. Check each stage output
4. Compare with ground truth

---

## Quick Tips - 快速提示

💡 **Tip 1:** To quickly test changes, use `detailed_trainer.py` with `num_samples: 1`

💡 **Tip 2:** Config supports both old and new formats - no need to update immediately

💡 **Tip 3:** Model code is now cleaner - easier to debug and maintain

💡 **Tip 4:** All logging is centralized in trainer files - better control

💡 **Tip 5:** Use `detailed_debug.print_prompts: true` to see exactly what LLM receives

---

For more details, see `CODE_OPTIMIZATION_SUMMARY.md`
