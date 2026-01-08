#!/bin/bash

# ============================================================================
# Hyperparameter Experiment Script
# 超参数实验脚本
# ============================================================================
# This script runs experiments with different combinations of:
# - rag.top_m_samples: 1, 3, 5, 10, 15
# - gravity.radius: 3, 5, 10, 15, 20
# ============================================================================

echo "======================================================================"
echo "Hyperparameter Experiment for Mobility Prediction"
echo "======================================================================"
echo ""

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/workspace/China_Journal"

# Configuration
BASE_CONFIG="/workspace/China_Journal/config/config.yaml"
RESULT_DIR="/workspace/China_Journal/result"
BACKUP_CONFIG="${BASE_CONFIG}.backup"
EXPERIMENT_LOG="${RESULT_DIR}/experiment_summary_$(date +%Y%m%d_%H%M%S).log"

# Create result directory if it doesn't exist
mkdir -p "$RESULT_DIR"

# Backup original config
cp "$BASE_CONFIG" "$BACKUP_CONFIG"
echo "Original config backed up to: $BACKUP_CONFIG"
echo ""

# Parameter values to test
RAG_VALUES=(1 3 5 10 15 20)
RADIUS_VALUES=(3 5 15 20)

# Default values when not being tested
DEFAULT_RAG=10
DEFAULT_RADIUS=10

# Total experiments: test RAG with fixed radius + test RADIUS with fixed RAG
TOTAL_EXPERIMENTS=$((${#RAG_VALUES[@]} + ${#RADIUS_VALUES[@]}))
CURRENT_EXPERIMENT=0

echo "======================================================================"
echo "Experiment Configuration:"
echo "  Part 1: Test RAG top_m_samples = ${RAG_VALUES[@]} with fixed radius = $DEFAULT_RADIUS"
echo "  Part 2: Test Gravity radius = ${RADIUS_VALUES[@]} with fixed RAG top_m = $DEFAULT_RAG"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Test file: data/beijing/small_test.csv"
echo "  Result directory: $RESULT_DIR"
echo "======================================================================"
echo ""

# Initialize summary log
echo "Hyperparameter Experiment Summary" > "$EXPERIMENT_LOG"
echo "Started: $(date)" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"
echo "" >> "$EXPERIMENT_LOG"
printf "%-5s %-12s %-12s %-10s %-10s %-10s %-10s %-10s\n" "Exp#" "RAG_top_m" "Radius" "Acc@1" "Acc@3" "Acc@5" "MRR@5" "ADE(km)" >> "$EXPERIMENT_LOG"
echo "--------------------------------------------------------------------------------" >> "$EXPERIMENT_LOG"

# Part 1: Test RAG values with fixed radius
echo "======================================================================"
echo "Part 1: Testing RAG top_m_samples (fixed radius = $DEFAULT_RADIUS)"
echo "======================================================================"
for rag_top_m in "${RAG_VALUES[@]}"; do
    gravity_radius=$DEFAULT_RADIUS
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "======================================================================"
        echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "  RAG top_m_samples = $rag_top_m"
        echo "  Gravity radius = $gravity_radius"
        echo "======================================================================"
        
        # Modify config.yaml using sed
        # Restore from backup first
        cp "$BACKUP_CONFIG" "$BASE_CONFIG"
        
        # Update RAG top_m_samples (line with "top_m_samples:")
        sed -i "s/top_m_samples: [0-9]*/top_m_samples: $rag_top_m/" "$BASE_CONFIG"
        
        # Update Gravity radius (line with "radius:")
        sed -i "s/radius: [0-9]*/radius: $gravity_radius/" "$BASE_CONFIG"
        
        # Update testing.num_samples to use all samples from small_test.csv
        # Set to a large number (10000) to ensure all samples are used
        sed -i "s/num_samples: .*/num_samples: 10000/" "$BASE_CONFIG"
        
        # Update data.city to beijing and ensure we're using the right test file
        sed -i 's/city: "[^"]*"/city: "beijing"/' "$BASE_CONFIG"
        
        echo "Config updated. Starting test..."
        
        # Create experiment-specific output directory
        EXP_DIR="${RESULT_DIR}/exp_${CURRENT_EXPERIMENT}_rag${rag_top_m}_radius${gravity_radius}"
        mkdir -p "$EXP_DIR"
        
        # Run test and capture output
        TEMP_LOG="${EXP_DIR}/test_output.log"
        python /workspace/China_Journal/trainer/trainer.py --config "$BASE_CONFIG" > "$TEMP_LOG" 2>&1
        
        # Check if test completed successfully
        if [ $? -eq 0 ]; then
            echo "✓ Test completed successfully"
            
            # Extract metrics from output
            # Looking for lines like:
            #   Acc@1: 0.1234
            #   Acc@3: 0.2345
            #   Acc@5: 0.3456
            #   MRR@5: 0.4567
            #   ADE: 1.2345 km
            
            ACC1=$(grep -oP "Acc@1: \K[0-9.]+" "$TEMP_LOG" | tail -1)
            ACC3=$(grep -oP "Acc@3: \K[0-9.]+" "$TEMP_LOG" | tail -1)
            ACC5=$(grep -oP "Acc@5: \K[0-9.]+" "$TEMP_LOG" | tail -1)
            MRR5=$(grep -oP "MRR@5: \K[0-9.]+" "$TEMP_LOG" | tail -1)
            ADE=$(grep -oP "ADE: \K[0-9.]+" "$TEMP_LOG" | tail -1)
            
            # Default to N/A if not found
            ACC1=${ACC1:-"N/A"}
            ACC3=${ACC3:-"N/A"}
            ACC5=${ACC5:-"N/A"}
            MRR5=${MRR5:-"N/A"}
            ADE=${ADE:-"N/A"}
            
            echo "Results:"
            echo "  Acc@1: $ACC1"
            echo "  Acc@3: $ACC3"
            echo "  Acc@5: $ACC5"
            echo "  MRR@5: $MRR5"
            echo "  ADE: $ADE km"
            
            # Append to summary log
            printf "%-5d %-12d %-12d %-10s %-10s %-10s %-10s %-10s\n" \
                "$CURRENT_EXPERIMENT" "$rag_top_m" "$gravity_radius" "$ACC1" "$ACC3" "$ACC5" "$MRR5" "$ADE" >> "$EXPERIMENT_LOG"
            
            # Save full output log
            mv "$TEMP_LOG" "${EXP_DIR}/test_output.log"
            
            # Copy detailed test logs if they exist
            LATEST_LOG=$(ls -t /workspace/China_Journal/output/logs/test_detailed_*.log 2>/dev/null | head -1)
            if [ -f "$LATEST_LOG" ]; then
                cp "$LATEST_LOG" "${EXP_DIR}/test_detailed.log"
            fi
            
        else
            echo "✗ Test failed"
            
            # Log failure
            printf "%-5d %-12d %-12d %-10s %-10s %-10s %-10s %-10s\n" \
                "$CURRENT_EXPERIMENT" "$rag_top_m" "$gravity_radius" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$EXPERIMENT_LOG"
            
            # Save error log
            mv "$TEMP_LOG" "${EXP_DIR}/error.log"
        fi
        
        echo "Results saved to: $EXP_DIR"
        
done

# Part 2: Test RADIUS values with fixed RAG
echo ""
echo "======================================================================"
echo "Part 2: Testing Gravity radius (fixed RAG top_m = $DEFAULT_RAG)"
echo "======================================================================"
for gravity_radius in "${RADIUS_VALUES[@]}"; do
    rag_top_m=$DEFAULT_RAG
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo ""
    echo "======================================================================"
    echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
    echo "  RAG top_m_samples = $rag_top_m"
    echo "  Gravity radius = $gravity_radius"
    echo "======================================================================"
    
    # Modify config.yaml using sed
    # Restore from backup first
    cp "$BACKUP_CONFIG" "$BASE_CONFIG"
    
    # Update RAG top_m_samples (line with "top_m_samples:")
    sed -i "s/top_m_samples: [0-9]*/top_m_samples: $rag_top_m/" "$BASE_CONFIG"
    
    # Update Gravity radius (line with "radius:")
    sed -i "s/radius: [0-9]*/radius: $gravity_radius/" "$BASE_CONFIG"
    
    # Update testing.num_samples to use all samples from small_test.csv
    # Set to a large number (10000) to ensure all samples are used
    sed -i "s/num_samples: .*/num_samples: 10000/" "$BASE_CONFIG"
    
    # Update data.city to beijing and ensure we're using the right test file
    sed -i 's/city: "[^"]*"/city: "beijing"/' "$BASE_CONFIG"
    
    echo "Config updated. Starting test..."
    
    # Create experiment-specific output directory
    EXP_DIR="${RESULT_DIR}/exp_${CURRENT_EXPERIMENT}_rag${rag_top_m}_radius${gravity_radius}"
    mkdir -p "$EXP_DIR"
    
    # Run test and capture output
    TEMP_LOG="${EXP_DIR}/test_output.log"
    python /workspace/China_Journal/trainer/trainer.py --config "$BASE_CONFIG" > "$TEMP_LOG" 2>&1
    
    # Check if test completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ Test completed successfully"
        
        # Extract metrics from output
        # Looking for lines like:
        #   Acc@1: 0.1234
        #   Acc@3: 0.2345
        #   Acc@5: 0.3456
        #   MRR@5: 0.4567
        #   ADE: 1.2345 km
        
        ACC1=$(grep -oP "Acc@1: \K[0-9.]+" "$TEMP_LOG" | tail -1)
        ACC3=$(grep -oP "Acc@3: \K[0-9.]+" "$TEMP_LOG" | tail -1)
        ACC5=$(grep -oP "Acc@5: \K[0-9.]+" "$TEMP_LOG" | tail -1)
        MRR5=$(grep -oP "MRR@5: \K[0-9.]+" "$TEMP_LOG" | tail -1)
        ADE=$(grep -oP "ADE: \K[0-9.]+" "$TEMP_LOG" | tail -1)
        
        # Default to N/A if not found
        ACC1=${ACC1:-"N/A"}
        ACC3=${ACC3:-"N/A"}
        ACC5=${ACC5:-"N/A"}
        MRR5=${MRR5:-"N/A"}
        ADE=${ADE:-"N/A"}
        
        echo "Results:"
        echo "  Acc@1: $ACC1"
        echo "  Acc@3: $ACC3"
        echo "  Acc@5: $ACC5"
        echo "  MRR@5: $MRR5"
        echo "  ADE: $ADE km"
        
        # Append to summary log
        printf "%-5d %-12d %-12d %-10s %-10s %-10s %-10s %-10s\n" \
            "$CURRENT_EXPERIMENT" "$rag_top_m" "$gravity_radius" "$ACC1" "$ACC3" "$ACC5" "$MRR5" "$ADE" >> "$EXPERIMENT_LOG"
        
        # Save full output log
        mv "$TEMP_LOG" "${EXP_DIR}/test_output.log"
        
        # Copy detailed test logs if they exist
        LATEST_LOG=$(ls -t /workspace/China_Journal/output/logs/test_detailed_*.log 2>/dev/null | head -1)
        if [ -f "$LATEST_LOG" ]; then
            cp "$LATEST_LOG" "${EXP_DIR}/test_detailed.log"
        fi
        
    else
        echo "✗ Test failed"
        
        # Log failure
        printf "%-5d %-12d %-12d %-10s %-10s %-10s %-10s %-10s\n" \
            "$CURRENT_EXPERIMENT" "$rag_top_m" "$gravity_radius" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$EXPERIMENT_LOG"
        
        # Save error log
        mv "$TEMP_LOG" "${EXP_DIR}/error.log"
    fi
    
    echo "Results saved to: $EXP_DIR"
    
done

# Restore original config
cp "$BACKUP_CONFIG" "$BASE_CONFIG"
rm "$BACKUP_CONFIG"
echo ""
echo "======================================================================"
echo "All experiments completed!"
echo "======================================================================"
echo "Original config restored: $BASE_CONFIG"
echo "Experiment summary: $EXPERIMENT_LOG"
echo ""

# Display summary
echo "======================================================================"
echo "Experiment Summary:"
echo "======================================================================"
cat "$EXPERIMENT_LOG"
echo ""
echo "======================================================================"
echo "Completed: $(date)"
echo "======================================================================"
