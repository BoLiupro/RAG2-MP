"""
Trainer script for Mobility Prediction with LLM+RAG
This script demonstrates how to use the LLM and RAG modules.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from model.LLM import MobilityLLM
from model.RAG import MobilityRAG
from typing import List, Dict, Any
from datetime import datetime


def load_trajectory_data(data_path: str, max_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Load and prepare trajectory data for training/testing.
    
    Args:
        data_path: Path to the CSV file (train.csv, val.csv, or test.csv)
        max_samples: Maximum number of samples to load (for testing)
    
    Returns:
        List of trajectory samples
    """
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Group by user_id to create trajectory samples
    samples = []
    
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp')
        
        # Create sliding window samples (obs_len=12, pred_len=1)
        obs_len = 12
        pred_len = 1
        window_size = obs_len + pred_len
        
        if len(group) < window_size:
            continue
        
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i:i + window_size]
            
            # Observation trajectory
            obs = window.iloc[:obs_len]
            trajectory = []
            for _, row in obs.iterrows():
                trajectory.append({
                    'location_id': row['location_id'],
                    'timestamp': row['timestamp'],
                    'user_id': user_id
                })
            
            # Next location (ground truth)
            next_location = window.iloc[obs_len]['location_id']
            
            samples.append({
                'trajectory': trajectory,
                'next_location': next_location,
                'user_id': user_id
            })
            
            # Limit samples for testing
            if len(samples) >= max_samples:
                break
        
        if len(samples) >= max_samples:
            break
    
    print(f"Loaded {len(samples)} trajectory samples")
    return samples


def test_llm_initialization():
    """Test LLM initialization."""
    print("\n" + "="*70)
    print("TEST 1: LLM Initialization")
    print("="*70)
    
    try:
        llm = MobilityLLM(
            model_name="Deepseek-R1-Distill-Qwen-3B",
            model_path="/datadisk",
            use_quantization=True,
            use_lora=False
        )
        print("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        return None


def test_trajectory_encoding(llm: MobilityLLM):
    """Test trajectory encoding."""
    print("\n" + "="*70)
    print("TEST 2: Trajectory Encoding")
    print("="*70)
    
    # Create a sample trajectory
    sample_trajectory = [
        {'location_id': 123, 'timestamp': '20220705 08:00'},
        {'location_id': 456, 'timestamp': '20220705 08:30'},
        {'location_id': 789, 'timestamp': '20220705 09:00'},
    ]
    
    try:
        embedding = llm.encode_trajectory(
            trajectory=sample_trajectory,
            city='beijing'
        )
        print(f"✓ Trajectory encoded successfully")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        return embedding
    except Exception as e:
        print(f"✗ Trajectory encoding failed: {e}")
        return None


def test_rag_initialization(llm: MobilityLLM, city: str = 'beijing'):
    """Test RAG module initialization."""
    print("\n" + "="*70)
    print("TEST 3: RAG Module Initialization")
    print("="*70)
    
    try:
        rag = MobilityRAG(
            llm=llm,
            rag_database_path="/workspace/China_Journal/model/rag_database",
            top_m=5,
            city=city
        )
        print("✓ RAG module initialized successfully")
        return rag
    except Exception as e:
        print(f"✗ RAG module initialization failed: {e}")
        return None


def test_rag_database_building(rag: MobilityRAG, city: str = 'beijing'):
    """Test RAG database building."""
    print("\n" + "="*70)
    print("TEST 4: RAG Database Building")
    print("="*70)
    
    # Load training data
    train_data_path = f"/workspace/China_Journal/data/{city}/train.csv"
    
    if not os.path.exists(train_data_path):
        print(f"✗ Training data not found at {train_data_path}")
        return False
    
    # Load POI data
    poi_data_path = f"/workspace/China_Journal/raw_data/{city}/{city}_grid_poi.csv"
    if os.path.exists(poi_data_path):
        rag.load_poi_data(poi_data_path)
    
    # Load a small subset of training samples
    training_samples = load_trajectory_data(train_data_path, max_samples=50)
    
    if len(training_samples) == 0:
        print("✗ No training samples loaded")
        return False
    
    try:
        # Build RAG database
        rag.build_rag_database(
            training_samples=training_samples,
            batch_size=10
        )
        print("✓ RAG database built successfully")
        rag.print_statistics()
        return True
    except Exception as e:
        print(f"✗ RAG database building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_retrieval(rag: MobilityRAG):
    """Test RAG retrieval and summary generation."""
    print("\n" + "="*70)
    print("TEST 5: RAG Retrieval and Summary Generation")
    print("="*70)
    
    # Create a query trajectory
    query_trajectory = [
        {'location_id': 100, 'timestamp': '20220706 08:00'},
        {'location_id': 150, 'timestamp': '20220706 08:30'},
        {'location_id': 200, 'timestamp': '20220706 09:00'},
    ]
    
    try:
        # Generate RAG summary
        summary, similar_samples, similarities = rag.generate_rag_summary(
            query_trajectory=query_trajectory,
            top_m=3
        )
        
        print("\n✓ RAG retrieval and summary generation successful")
        print(f"\nGenerated Summary:")
        print(f"{'-'*70}")
        print(summary)
        print(f"{'-'*70}")
        
        return summary
    except Exception as e:
        print(f"✗ RAG retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_next_location_prediction(llm: MobilityLLM, rag_summary: str = None):
    """Test next location prediction."""
    print("\n" + "="*70)
    print("TEST 6: Next Location Prediction")
    print("="*70)
    
    query_trajectory = [
        {'location_id': 100, 'timestamp': '20220706 08:00'},
        {'location_id': 150, 'timestamp': '20220706 08:30'},
        {'location_id': 200, 'timestamp': '20220706 09:00'},
    ]
    
    candidate_locations = [250, 300, 350, 400, 450]
    
    try:
        predictions = llm.predict_next_location(
            query_trajectory=query_trajectory,
            candidate_locations=candidate_locations,
            rag_summary=rag_summary,
            city='beijing',
            top_k=3
        )
        
        print("✓ Next location prediction successful")
        print(f"\nTop-3 Predictions:")
        for i, (loc_id, confidence) in enumerate(predictions, 1):
            print(f"  {i}. Location {loc_id} (confidence: {confidence:.4f})")
        
        return predictions
    except Exception as e:
        print(f"✗ Next location prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run all tests."""
    print("\n" + "="*70)
    print("Mobility Prediction LLM+RAG Framework Testing")
    print("="*70)
    
    # Test 1: Initialize LLM
    llm = test_llm_initialization()
    if llm is None:
        print("\nFailed to initialize LLM. Exiting.")
        return
    
    # Test 2: Encode trajectory
    embedding = test_trajectory_encoding(llm)
    if embedding is None:
        print("\nFailed to encode trajectory. Exiting.")
        return
    
    # Test 3: Initialize RAG
    city = 'beijing'  # Change to 'shenzhen' or 'nanchang' as needed
    rag = test_rag_initialization(llm, city=city)
    if rag is None:
        print("\nFailed to initialize RAG. Exiting.")
        return
    
    # Test 4: Build RAG database (optional - takes time)
    build_database = input("\nBuild RAG database? (y/n): ").lower() == 'y'
    if build_database:
        success = test_rag_database_building(rag, city=city)
        if not success:
            print("\nFailed to build RAG database. Continuing with other tests.")
    
    # Test 5: RAG retrieval (only if database exists)
    if rag.database_embeddings is not None:
        rag_summary = test_rag_retrieval(rag)
    else:
        print("\nSkipping RAG retrieval test (database not built)")
        rag_summary = None
    
    # Test 6: Next location prediction
    test_next_location_prediction(llm, rag_summary)
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
