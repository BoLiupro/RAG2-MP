#!/bin/bash
# Script to fit gravity model weights for each city

# Default parameters
CITY="beijing"
DATA_DIR="/workspace/China_Journal/data"
OUTPUT_DIR="/workspace/China_Journal/util/gravity_weight"
OBS_LEN=12
RADIUS=10
MAX_SAMPLES=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --city)
            CITY="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --obs_len)
            OBS_LEN="$2"
            shift 2
            ;;
        --radius)
            RADIUS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Fitting Gravity Model Weights"
echo "============================================"
echo "City: $CITY"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Observation length: $OBS_LEN"
echo "Radius: $RADIUS"
echo "Max samples: $MAX_SAMPLES"
echo "============================================"
echo ""

# Run the weight fitting script
python /workspace/China_Journal/util/fit_gravity_weight.py \
    --city "$CITY" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --obs_len "$OBS_LEN" \
    --radius "$RADIUS" \
    --max_samples "$MAX_SAMPLES"

echo ""
echo "============================================"
echo "Weight fitting completed!"
echo "Results saved to: $OUTPUT_DIR/gravity_weights_${CITY}.json"
echo "============================================"
echo ""
echo "To use these weights in training, update config.yaml:"
echo "  model:"
echo "    gravity:"
echo "      weight_config_path: \"$OUTPUT_DIR/gravity_weights_${CITY}.json\""
