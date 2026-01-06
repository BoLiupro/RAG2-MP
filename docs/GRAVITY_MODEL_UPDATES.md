# Gravity Model Weight Fitting and Score Smoothing

## Overview

This update adds several key improvements to the Gravity Model:

1. **Per-Category Weight Fitting**: Automatically fit optimal weight parameters for each POI category based on training data
2. **Score Smoothing**: Normalize gravity scores to [0, 1] range to prevent extreme values
3. **Sample Filtering**: Filter out staying-in-place samples during weight fitting
4. **Special Score Handling**: Fixed moderate score for staying in same location to avoid division-by-near-zero

## Recent Updates (2026-01-06)

### 🔧 Critical Fixes

1. **Sample Filtering for Weight Fitting**
   - Staying-in-place samples (where `current_location == next_location`) are now **excluded** from weight fitting
   - These samples don't provide meaningful information for fitting movement weights
   - Filtering happens in `_create_samples()` method

2. **Special Handling for Same-Location Scores**
   - When `current_grid_id == target_grid_id`, score is set to **`weight × 10.0`** (fixed moderate value)
   - Prevents score "explosion" from dividing by very small distance
   - Represents "inertia" of staying in current location
   - Applied in both `find_gravity_coef.py` and `Gravity.py`

3. **Updated Save Path**
   - Weights now saved to: **`/workspace/China_Journal/util/gravity_weight/`**
   - Previously: `/workspace/China_Journal/config/`
   - Better organization: keeps utility-generated files in util directory

## Changes Made

### 1. Modified Files

- **`/workspace/China_Journal/model/Gravity.py`**
  - Added support for per-category weights (dictionary or single float)
  - Added `_smooth_score()` function for min-max normalization
  - Modified `_calculate_gravity_score()` to use category-specific weights
  - Modified `get_candidate_locations()` to apply smoothing
  - Added `weight_config_path` parameter to load fitted weights from JSON

- **`/workspace/China_Journal/util/find_gravity_coef.py`**
  - Created new script to fit optimal weights for each POI category
  - Implements `GravityWeightFitter` class
  - Uses scipy optimization to minimize average ranking error
  - Saves fitted weights to JSON file

- **`/workspace/China_Journal/model/Predictor.py`**
  - Added `gravity_weight_config_path` parameter
  - Passes config path to GravityModel initialization

- **`/workspace/China_Journal/trainer/trainer.py`**
  - Added support for `weight_config_path` from config
  - Logs weight configuration during initialization

- **`/workspace/China_Journal/config/config.yaml`**
  - Added `weight_config_path` parameter (optional)
  - Added documentation for weight fitting

- **`/workspace/China_Journal/scripts/fit_gravity_weights.sh`**
  - Created convenience script to run weight fitting

## How to Use

### Step 1: Fit Gravity Weights (Optional)

If you want to use data-driven weights instead of default weights:

```bash
# Fit weights for a specific city
cd /workspace/China_Journal
bash scripts/fit_gravity_weights.sh --city beijing --max_samples 1000

# Or use the Python script directly
python util/find_gravity_coef.py \
    --city beijing \
    --data_dir /workspace/China_Journal/data \
    --output_dir /workspace/China_Journal/util/gravity_weight \
    --obs_len 12 \
    --radius 10 \
    --max_samples 1000
```

This will:
- Read training data from `data/{city}/train.csv`
- **Filter out staying-in-place samples** (where user doesn't move)
- Fit optimal weight for each of 14 POI categories
- Save results to `util/gravity_weight/gravity_weights_{city}.json`
- Print evaluation metrics

### Step 2: Configure Training

Update `config/config.yaml` to use fitted weights:

```yaml
model:
  gravity:
    top_n_candidates: 3
    weight: 1.0  # Default weight (used if weight_config_path is null or file not found)
    radius: 5
    grid_size: 40
    weight_config_path: "/workspace/China_Journal/util/gravity_weight/gravity_weights_beijing.json"
```

If `weight_config_path` is:
- **null or file doesn't exist**: Uses default `weight` value (1.0) for all categories
- **Valid path to JSON**: Loads per-category weights from file

### Step 3: Train Model

```bash
cd /workspace/China_Journal
bash scripts/train.sh
```

The model will automatically use the fitted weights if configured.

## Weight Fitting Algorithm

The weight fitting process:

1. **Sample Creation**: Creates trajectory samples from training data (obs_len + pred_len)
   - **Filters out staying-in-place samples** where `current_location == next_location`
   - Only uses samples where user actually moves to a different location
2. **Optimization**: For each POI category, optimizes weight parameter to minimize average ranking of true next location
3. **Objective Function**: Lower average ranking = better weights (ideally rank #1)
4. **Method**: Uses Nelder-Mead optimization (scipy.optimize.minimize)
5. **Special Handling**: When calculating scores for same location, uses fixed value `weight × 10.0` instead of distance-based formula

**Formula**: 
- Movement: `score = weight_of_category * POI_count / distance^2`
- Staying: `score = weight_of_category * 10.0` (fixed)

## Score Smoothing

All gravity scores are now smoothed using min-max normalization:

```
smoothed_score = (raw_score - min_score) / (max_score - min_score)
```

Benefits:
- Prevents extreme scores when distance is very small
- Normalizes scores to [0, 1] range (3 decimal places)
- Makes scores comparable across different categories
- Improves numerical stability

## Output Format

### Fitted Weights JSON

Saved to: `/workspace/China_Journal/util/gravity_weight/gravity_weights_{city}.json`

```json
{
  "city": "beijing",
  "weights": {
    "Transportation Facilities_count": 1.5234,
    "Dining & Cuisine_count": 1.2345,
    ...
  },
  "evaluation": {
    "Transportation Facilities_count": 12.34,
    ...
  },
  "metadata": {
    "obs_len": 12,
    "radius": 10,
    "num_samples": 5000,
    "num_training_records": 41984
  }
}
```

### Candidate Locations with Scores

When `return_scores=True` in `get_candidate_locations()`:

```python
{
  "Transportation Facilities_count": [
    (516, 1.000),  # (grid_id, smoothed_score)
    (517, 0.253),
    (515, 0.145)
  ],
  ...
}
```

## Testing

Test the implementation:

```bash
cd /workspace/China_Journal
python test_gravity_updates.py
```

This will:
- Test score smoothing with default weights
- Test per-category custom weights
- Print sample candidate locations with smoothed scores
- Verify all features work correctly

## Performance Considerations

**Weight Fitting**:
- Can be slow for large datasets
- Use `--max_samples` to limit sample size (default: 1000)
- Only needs to be run once per city
- Results can be reused across multiple training runs

**Score Smoothing**:
- Minimal computational overhead
- Applied during candidate selection
- No impact on training/inference speed

## Backward Compatibility

The changes are fully backward compatible:

- If `weight_config_path` is not specified, uses default weight
- If `weight` is a single float, applies same weight to all categories
- Old configs will continue to work without modification

## Next Steps

1. **Fit weights for all cities**: Run weight fitting for beijing, nanchang, and shenzhen
2. **Evaluate impact**: Compare model performance with/without fitted weights
3. **Fine-tune parameters**: Adjust `max_samples`, `radius` if needed
4. **Production use**: Use fitted weights in final training runs

## Troubleshooting

**Issue**: Weight fitting is slow
- **Solution**: Reduce `--max_samples` parameter

**Issue**: Fitted weights seem unreasonable
- **Solution**: Check data quality, increase sample size, verify POI data

**Issue**: Config not loading weights
- **Solution**: Verify file path, check JSON format, ensure file exists

**Issue**: Scores all the same after smoothing
- **Solution**: Normal if all raw scores are similar; indicates low variation in gravity scores
