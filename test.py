"""
Standalone test script for MobilityPredictor
Tests the complete prediction pipeline with a simple example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

from model.Predictor import MobilityPredictor


def test_basic_prediction():
    """Test basic prediction functionality."""
    print("\n" + "="*70)
    print("Testing MobilityPredictor - Basic Prediction")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = MobilityPredictor(
        llm_model_name="Deepseek-R1-Distill-Qwen-3B",
        llm_model_path="/datadisk",
        city="beijing",
        top_k_predictions=5,
        rag_top_m_samples=3,
        gravity_top_n_candidates=5,
        gravity_weight=1.0,
        gravity_radius=10,
        use_quantization=True,
        verbose=True
    )
    
    # Test trajectory
    trajectory = [
        {'location_id': 780, 'timestamp': '20220706 08:00'},
        {'location_id': 781, 'timestamp': '20220706 08:30'},
        {'location_id': 800, 'timestamp': '20220706 09:00'},
    ]
    
    ground_truth = 802
    
    # Make prediction
    predictions, results = predictor.predict(
        observation_trajectory=trajectory,
        ground_truth=ground_truth,
        print_prompt=True
    )
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70 + "\n")
    
    return predictions, results


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    test_basic_prediction()
