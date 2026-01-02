# RAG Database Building Guide

## Overview
The `build_rag_database.py` script builds the RAG (Retrieval-Augmented Generation) experience pool database from training data. This database is required before training the mobility prediction model.

## What It Does
1. Loads training samples from the dataset
2. Encodes each sample using the LLM to create embeddings
3. Builds a FAISS index for efficient similarity search
4. Saves the database to disk for use during training and inference

## Usage

### Basic Usage
```bash
cd /workspace/China_Journal
python util/build_rag_database.py --config config.yaml
```

### Override City
```bash
python util/build_rag_database.py --config config.yaml --city beijing
```

### Override Batch Size
```bash
python util/build_rag_database.py --config config.yaml --batch-size 16
```

### Force Overwrite Existing Database
```bash
python util/build_rag_database.py --config config.yaml --overwrite
```

## Configuration

The script uses parameters from `config.yaml`:

### Data Configuration
```yaml
data:
  city: 'beijing'
  data_dir: './data'
  obs_len: 12
  pred_len: 1
  max_train_samples: 10000  # Limit training samples (optional)
```

### RAG Build Configuration
```yaml
rag_build:
  output_dir: './util/rag_database'
  batch_size: 32
  save_faiss_index: true
  overwrite: false
```

### Model Configuration
```yaml
model:
  llm_model_name: 'Deepseek-R1-Distill-Qwen-3B'
  llm_model_path: '/datadisk/Deepseek-R1-Distill-Qwen-3B'
  use_quantization: true
  rag_top_m_samples: 5  # Number of similar samples to retrieve
```

## Output Structure

The script creates the following structure:
```
util/rag_database/
└── {city}/
    ├── embeddings.npy           # Sample embeddings
    ├── samples.pkl               # Sample metadata
    ├── faiss_index.bin          # FAISS index
    └── metadata.json            # Build metadata
```

## Example Workflow

### 1. Prepare Data
Ensure your data is organized as:
```
data/
└── {city}/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── poi.csv (optional)
```

### 2. Configure Parameters
Edit `config.yaml`:
```yaml
data:
  city: 'beijing'
  max_train_samples: 10000

rag_build:
  batch_size: 32
  overwrite: false
```

### 3. Build RAG Database
```bash
python util/build_rag_database.py --config config.yaml
```

### 4. Verify Database
The script will automatically verify the database after building and print statistics:
```
RAG Database Statistics:
  Number of samples: 10000
  Embedding dimension: 4096
  FAISS index type: IndexFlatIP
  FAISS index size: 10000
```

### 5. Start Training
Once the RAG database is built, you can start training:
```bash
bash scripts/train.sh config.yaml
```

## Performance Tips

### Memory Optimization
- Use smaller `batch_size` (16-32) if you encounter OOM errors
- Enable `use_quantization: true` to reduce memory usage
- Limit `max_train_samples` for faster builds during development

### Speed Optimization
- Use larger `batch_size` (64-128) if memory allows
- Build on GPU for faster encoding
- Use SSD storage for faster I/O

## Troubleshooting

### Error: "RAG database already exists"
Solution: Use `--overwrite` flag or set `overwrite: true` in config

### Error: "CUDA out of memory"
Solution: Reduce `batch_size` in config or use `--batch-size 16`

### Error: "POI data not found"
Solution: This is a warning. POI data is optional. The script will continue without it.

### Error: "Failed to verify RAG database"
Solution: Check output directory permissions and disk space

## Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Path to config.yaml (required) | `--config config.yaml` |
| `--city` | Override city from config | `--city beijing` |
| `--batch-size` | Override batch size | `--batch-size 16` |
| `--overwrite` | Overwrite existing database | `--overwrite` |

## Notes

- **Required Before Training**: You must build the RAG database before training
- **One Database Per City**: Build a separate database for each city
- **Rebuilding**: If you change training data, rebuild the database
- **Storage**: Each database requires ~50-200MB depending on sample count
