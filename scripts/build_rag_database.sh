#!/bin/bash

# Build RAG Database Script
# This script builds the experience pool database for RAG-based mobility prediction

echo "======================================================================"
echo "Building RAG Database for Mobility Prediction Model"
echo "======================================================================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/workspace/China_Journal"

# Default parameters
CONFIG_FILE="${1:-/workspace/China_Journal/config/config.yaml}"
CITY="${2:-beijing}"
TARGET_CLUSTERS="${3:-100}"
SAMPLES_PER_CLUSTER="${4:-10}"
# MAX_TRAIN_SAMPLES="${5:-500}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [city] [target_clusters] [samples_per_cluster] [max_train_samples]"
    exit 1
fi

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  City: $CITY"
echo "  Target clusters: $TARGET_CLUSTERS"
echo "  Samples per cluster: $SAMPLES_PER_CLUSTER"
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    echo "  Max train samples: $MAX_TRAIN_SAMPLES"
else
    echo "  Max train samples: all"
fi
echo ""

# Build command
CMD="python /workspace/China_Journal/util/build_rag_database.py --config $CONFIG_FILE --city $CITY --target_clusters $TARGET_CLUSTERS --samples_per_cluster $SAMPLES_PER_CLUSTER"

if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    CMD="$CMD --max_train_samples $MAX_TRAIN_SAMPLES"
fi

# Run database building
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "======================================================================"
echo "RAG Database build completed!"
echo "======================================================================"
