#!/bin/bash

# Training script for Mobility Prediction Model
# Reads configuration from config.yaml and starts training

echo "======================================================================"
echo "Mobility Prediction Model Training Detailed Version"
echo "======================================================================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/workspace/China_Journal"

# Default config file
CONFIG_FILE="${1:-/workspace/China_Journal/config/config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""

# Run training
python /workspace/China_Journal/trainer/detailed_trainer.py --config "$CONFIG_FILE"

echo ""
echo "======================================================================"
echo "Detailed training completed!"
echo "======================================================================"
