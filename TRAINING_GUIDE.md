# Training Pipeline Documentation

## Overview

This document describes the complete training pipeline for the Mobility Prediction Model, including data loading, model training, validation, testing, and logging.

## File Structure

```
/workspace/China_Journal/
├── config.yaml                 # Configuration file for all parameters
├── dataset/
│   └── dataset.py             # Dataset loading and preprocessing
├── trainer/
│   └── trainer.py             # Complete training pipeline
├── scripts/
│   └── train.sh               # Training launch script
├── model/
│   ├── LLM.py                 # LLM module
│   ├── RAG.py                 # RAG module
│   ├── Gravity.py             # Gravity model
│   └── Predictor.py           # Main predictor
├── common/
│   └── utils.py               # Utility functions
└── output/                    # Training outputs (created automatically)
    ├── checkpoints/           # Model checkpoints
    └── logs/                  # Training logs
```

## Configuration (config.yaml)

All training parameters are managed through `config.yaml`:

### Data Configuration
- `city`: City name (beijing, shenzhen, nanchang)
- `data_dir`: Base directory for data
- `obs_len`: Observation trajectory length (default: 12)
- `pred_len`: Prediction trajectory length (default: 1)
- `max_train_samples`: Maximum training samples (null = all)
- `max_val_samples`: Maximum validation samples (null = all)
- `max_test_samples`: Maximum test samples (null = all)

### Model Configuration
- `llm_model_name`: LLM model name
- `llm_model_path`: Path to LLM model
- `top_k_predictions`: Number of predictions to return
- `rag_top_m_samples`: Similar samples to retrieve
- `gravity_top_n_candidates`: Candidates per POI category
- `gravity_weight`: Gravity formula weight
- `gravity_radius`: Search radius in grids
- `use_quantization`: Enable 4-bit quantization

### Training Configuration
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay for regularization
- `max_grad_norm`: Gradient clipping threshold
- `samples_per_epoch`: Samples per epoch (null = all)
- `val_every_n_epochs`: Validation frequency
- `save_every_n_epochs`: Checkpoint save frequency
- `output_dir`: Output directory

## Dataset (dataset.py)

### MobilityDataset Class

Loads and preprocesses trajectory data:

```python
dataset = MobilityDataset(
    data_path="data/beijing/train.csv",
    city="beijing",
    obs_len=12,
    pred_len=1,
    max_samples=None,
    poi_data_path="data/beijing/poi.csv"
)
```

**Features:**
- Loads CSV files with user trajectories
- Creates sliding window samples
- Supports POI data integration
- Provides dataset statistics

**Data Format:**
- Input CSV: `user_id, timestamp, location_id`
- Sample structure:
  ```python
  {
      'user_id': int,
      'observation': [{'location_id': int, 'timestamp': str}, ...],
      'prediction': [{'location_id': int, 'timestamp': str}, ...],
      'next_location': int
  }
  ```

## Training Pipeline (trainer.py)

### MobilityTrainer Class

Complete training workflow:

1. **Data Loading**: Loads train/val/test datasets
2. **Model Initialization**: Initializes MobilityPredictor
3. **Training Loop**: 
   - Forward pass through RAG + Gravity + LLM
   - Loss computation (cross-entropy)
   - Backward pass and optimization
4. **Validation**: Evaluates on validation set
5. **Testing**: Final evaluation with detailed logging
6. **Checkpointing**: Saves best models

### Loss Function

The loss is computed as cross-entropy between:
- LLM output logits (after prompt)
- Ground truth text: `"Grid {location_id}"`

```python
loss = cross_entropy(
    llm_logits[prompt_length:],
    ground_truth_tokens
)
```

### Evaluation Metrics

- **Accuracy@K**: Proportion of samples where ground truth is in top-K predictions (K=1,3,5,10)
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for ground truth

## Usage

### 1. Test Setup

```bash
python test_training_setup.py
```

This validates:
- Dataset loading
- Configuration parsing
- Basic functionality

### 2. Start Training

**Option A: Using shell script**
```bash
bash scripts/train.sh
# or with custom config
bash scripts/train.sh /path/to/custom_config.yaml
```

**Option B: Direct Python**
```bash
python trainer/trainer.py --config config.yaml
```

### 3. Monitor Training

Training progress is logged to:
- Console (real-time with tqdm progress bars)
- Log file: `output/logs/training_YYYYMMDD_HHMMSS.log`
- Metrics file: `output/logs/metrics.json`

### 4. Checkpoints

Saved to `output/checkpoints/`:
- `checkpoint_epoch_N.pt`: Regular checkpoints
- `best_model.pt`: Best model based on Val Acc@5

## Output Files

### Training Log
```
[2026-01-03 10:00:00] Loading datasets...
[2026-01-03 10:01:00] Train samples: 1000
[2026-01-03 10:01:00] Validation samples: 200
[2026-01-03 10:01:00] Test samples: 200
[2026-01-03 10:02:00] Initializing MobilityPredictor model...
[2026-01-03 10:05:00] Model initialized successfully
[2026-01-03 10:05:00] STARTING TRAINING
[2026-01-03 10:05:00] Epoch 1/10
[2026-01-03 10:15:00] Train Loss: 2.3456
[2026-01-03 10:18:00] Val Loss: 2.1234
[2026-01-03 10:18:00] Val Acc@1: 0.1200
[2026-01-03 10:18:00] Val Acc@5: 0.3500
...
```

### Detailed Test Log
```
======================================================================
TEST SAMPLE 1
======================================================================
Ground Truth: Grid 1234
Predictions: [1234, 1235, 1236, ...]
Loss: 1.2345

RAG Summary:
Based on 5 similar patterns...

Gravity Candidates:
  Shopping: [(1234, 0.85), (1235, 0.78), ...]
  Dining: [(1240, 0.92), (1241, 0.88), ...]
  ...

LLM Prediction Prompt:
You are analyzing mobility patterns...
[Full prompt text]

LLM Response:
Based on the analysis, the most likely next locations are...
======================================================================
```

### Metrics JSON
```json
{
  "train_losses": [2.34, 2.12, 1.98, ...],
  "val_metrics": [
    {
      "epoch": 1,
      "val_loss": 2.12,
      "val_acc@1": 0.12,
      "val_acc@3": 0.28,
      "val_acc@5": 0.35,
      "val_acc@10": 0.52,
      "val_mrr": 0.22
    },
    ...
  ],
  "best_val_acc": 0.45
}
```

## Training Notes

### Memory Management
- Use `max_train_samples` to limit memory usage during development
- 4-bit quantization reduces memory by ~75%
- Gradient accumulation (if needed) can be added

### Training Time
- 1 epoch with 500 samples: ~10-20 minutes (depending on hardware)
- Full training (10 epochs): ~2-3 hours

### Best Practices
1. Start with small `max_samples` values for testing
2. Monitor validation metrics to detect overfitting
3. Use `val_every_n_epochs=1` initially
4. Save checkpoints frequently during early experiments
5. Review detailed test logs to understand model behavior

## Troubleshooting

### Out of Memory
- Reduce `samples_per_epoch`
- Enable quantization
- Reduce `rag_top_m_samples`
- Reduce `obs_len`

### Poor Performance
- Increase `num_epochs`
- Adjust `learning_rate`
- Increase `rag_top_m_samples`
- Check data quality in logs

### Slow Training
- Reduce `samples_per_epoch`
- Use fewer validation samples
- Disable detailed logging during training

## Next Steps

After training:
1. Review test metrics in logs
2. Analyze detailed test logs for failure cases
3. Tune hyperparameters based on results
4. Run ablation studies (w/o RAG, w/o Gravity, w/o LLM predictor)
5. Test on other cities

## Contact

For questions or issues, refer to the main README.md or project documentation.
