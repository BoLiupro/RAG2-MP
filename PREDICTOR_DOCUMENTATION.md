# MobilityPredictor Module Documentation

## Overview

The `MobilityPredictor` class is the main controller that integrates all modules (LLM, RAG, and Gravity Model) to provide end-to-end mobility prediction.

## Architecture

```
┌─────────────────────────────────────────────┐
│         MobilityPredictor                    │
│  (Main Controller & Orchestrator)            │
└─────────────────────────────────────────────┘
          │
          ├──────────────┬────────────────┐
          │              │                │
          ▼              ▼                ▼
    ┌─────────┐    ┌─────────┐    ┌──────────┐
    │   LLM   │    │   RAG   │    │ Gravity  │
    │ Module  │    │ Module  │    │  Model   │
    └─────────┘    └─────────┘    └──────────┘
```

## Key Features

### 1. **Integrated Pipeline**
- Seamlessly combines LLM+RAG and Gravity Model
- Single interface for complete prediction workflow
- Automatic module initialization and coordination

### 2. **Configurable Parameters**
All hyperparameters can be set during initialization:
- `top_k`: Number of predictions to return (default: 5)
- `top_m`: Number of similar samples for RAG (default: 5)
- `top_n`: Number of candidates per POI category (default: 5)
- `gravity_weight`: Gravity model weight parameter (default: 1.0)
- `gravity_radius`: Search radius in grid units (default: 10)

### 3. **Verbose Mode**
- Prints all prompts sent to LLM
- Shows intermediate results (RAG summary, candidates)
- Displays prediction confidence scores

## Usage

### Basic Usage

```python
from model.MobilityPredictor import MobilityPredictor

# Initialize predictor
predictor = MobilityPredictor(
    llm_model_name="Deepseek-R1-Distill-Qwen-3B",
    city="beijing",
    top_k=5,
    top_m=3,
    top_n=5,
    verbose=True
)

# Prepare observation trajectory
trajectory = [
    {'location_id': 100, 'timestamp': '20220706 08:00'},
    {'location_id': 150, 'timestamp': '20220706 08:30'},
    {'location_id': 200, 'timestamp': '20220706 09:00'},
]

# Make prediction
predictions, results = predictor.predict(
    observation_trajectory=trajectory,
    ground_truth=250  # Optional, for evaluation
)

# Access results
for i, (loc_id, confidence) in enumerate(predictions, 1):
    print(f"{i}. Grid {loc_id} - Confidence: {confidence:.4f}")
```

### Batch Prediction

```python
# Multiple trajectories
trajectories = [traj1, traj2, traj3]
ground_truths = [gt1, gt2, gt3]

results = predictor.batch_predict(
    trajectories=trajectories,
    ground_truths=ground_truths
)
```

### Evaluation

```python
# Evaluate on test set
metrics = predictor.evaluate(
    trajectories=test_trajectories,
    ground_truths=test_labels
)

print(f"Accuracy@1: {metrics['accuracy@1']:.4f}")
print(f"Accuracy@5: {metrics['accuracy@5']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")
```

## Prediction Pipeline

### Step 1: RAG Retrieval
```
Input: Observation trajectory
↓
LLM encodes trajectory → embedding
↓
FAISS searches RAG database → top-m similar samples
↓
LLM generates summary → "User likely going to dining/shopping area"
```

### Step 2: Gravity Model
```
Input: Current location (last point in trajectory)
↓
For each POI category:
  - Compute gravity scores for nearby grids
  - Select top-n candidates
↓
Output: 14 groups × n candidates per group
```

### Step 3: Final Prediction
```
Input: RAG summary + Gravity candidates
↓
LLM analyzes user intent from summary
↓
LLM selects appropriate POI categories
↓
LLM ranks candidates → top-k predictions
```

## Output Format

### Predictions
List of `(location_id, confidence)` tuples:
```python
[
    (802, 1.0),      # Rank 1, highest confidence
    (804, 0.5),      # Rank 2
    (800, 0.333),    # Rank 3
    (760, 0.25),     # Rank 4
    (803, 0.2)       # Rank 5
]
```

### Results Dictionary
```python
{
    'predictions': [(loc_id, conf), ...],
    'rag_summary': "Text summary from RAG",
    'similar_samples': [sample1, sample2, ...],
    'similarities': [0.95, 0.92, 0.89],
    'candidates_by_category': {
        'Dining & Cusine_count': [(802, 500.0), ...],
        'Shopping & Consumer Goods_count': [(800, 1400.0), ...],
        ...
    },
    'current_location': 800,
    'observation_trajectory': [...]
}
```

## Prompt Design

### Final Prediction Prompt Structure

```
You are a mobility prediction expert analyzing [mobility_mode] patterns.
Your task is to predict the next location based on the following information:

## Current Trajectory:
1. Location 100 at 20220706 08:00
2. Location 150 at 20220706 08:30
3. Location 200 at 20220706 09:00

## Similar Historical Patterns:
[RAG summary text]

## Candidate Locations by POI Category:
Based on the gravity model, here are the most attractive locations:
- Dining & Cusine: Grid 802, Grid 760, Grid 804
- Shopping & Consumer Goods: Grid 800, Grid 802, Grid 804
- Transportation Facilities: Grid 800, Grid 760, Grid 801
...

## Your Task:
Based on the trajectory pattern, similar historical behaviors, and candidate locations,
predict the top 5 most likely next locations.
Choose from these candidates: Grid 800, Grid 802, Grid 760, ...

Format your answer as: Grid [ID], Grid [ID], Grid [ID], ...
Provide exactly 5 predictions in order of likelihood.

Your predictions:
```

## Evaluation Metrics

### Accuracy@K
Percentage of test cases where ground truth is in top-K predictions:
- **Accuracy@1**: Top-1 hit rate (strictest metric)
- **Accuracy@3**: Top-3 hit rate
- **Accuracy@5**: Top-5 hit rate

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks:
```
MRR = (1/N) × Σ(1/rank_i)
```
- rank_i = position of ground truth in predictions
- Higher is better (max = 1.0)

## Configuration Examples

### Conservative (High Precision)
```python
predictor = MobilityPredictor(
    top_k=3,           # Only top-3 predictions
    top_m=3,           # Few similar samples
    top_n=3,           # Few candidates per category
    gravity_radius=5,  # Small search area
    verbose=True
)
```

### Aggressive (High Recall)
```python
predictor = MobilityPredictor(
    top_k=10,          # More predictions
    top_m=10,          # More similar samples
    top_n=10,          # More candidates per category
    gravity_radius=15, # Larger search area
    verbose=True
)
```

### Balanced (Recommended)
```python
predictor = MobilityPredictor(
    top_k=5,
    top_m=5,
    top_n=5,
    gravity_radius=10,
    verbose=True
)
```

## Performance Considerations

### Memory Usage
- LLM: ~3-4GB (with 4-bit quantization)
- RAG Database: ~200MB per 10,000 samples
- FAISS Index: ~100MB per 10,000 samples

### Inference Time
- RAG Retrieval: ~50ms (with FAISS)
- Gravity Model: ~10ms
- LLM Generation: ~2-5s (depends on prompt length)
- **Total: ~3-6s per prediction**

### Optimization Tips
1. Use FAISS GPU for faster retrieval (if available)
2. Reduce `top_m` to decrease RAG overhead
3. Reduce `gravity_radius` to speed up candidate generation
4. Use smaller LLM models for faster inference

## Ablation Studies

### w/o RAG (Without RAG Module)
Set `top_m=0` and use empty summary:
```python
# Modify predict() to skip RAG
predictions = predictor._generate_final_prediction(
    observation_trajectory=trajectory,
    rag_summary="",  # Empty summary
    candidates_by_category=candidates
)
```

### w/o Gravity (Without Gravity Model)
Use all grid locations as candidates instead of gravity-filtered ones.

### w/o LLM Predictor (Direct Ranking)
Use gravity scores directly to rank candidates instead of LLM.

## Testing

### Unit Test
```bash
python test_predictor.py
```

### Full Test Suite
```bash
python trainer/trainer.py
```

## Troubleshooting

### Issue: RAG database not loaded
**Solution**: Build RAG database first using trainer script

### Issue: Out of memory
**Solution**: 
- Reduce batch size
- Use smaller LLM model
- Enable CPU offloading

### Issue: Slow predictions
**Solution**:
- Check if FAISS is using CPU (consider GPU version)
- Reduce prompt length
- Decrease `top_m` and `gravity_radius`

## Future Enhancements

1. **Multi-step Prediction**: Predict trajectory sequences instead of single next location
2. **Fine-tuning**: Train LLM with mobility-specific data
3. **Ensemble Methods**: Combine multiple prediction strategies
4. **Real-time Updates**: Dynamically update RAG database with new trajectories
5. **Context-aware Weighting**: Adjust gravity weights based on time/user profile

## References

- LLM Module: [model/LLM.py](model/LLM.py)
- RAG Module: [model/RAG.py](model/RAG.py)
- Gravity Model: [model/Gravity.py](model/Gravity.py)
- Test Script: [test_predictor.py](test_predictor.py)
