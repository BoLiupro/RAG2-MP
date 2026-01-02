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
from model.Gravity import GravityModel
from model.Predictor import MobilityPredictor
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
            rag_database_path="/workspace/China_Journal/util/rag_database",
            rag_top_m_samples=5,
            city=city,
            verbose=True  # Enable verbose mode for prompt printing
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
            rag_top_m_samples=3
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
            top_k_predictions=3
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


def test_gravity_model(city: str = 'beijing'):
    """Test Gravity Model for candidate location selection."""
    print("\n" + "="*70)
    print("TEST 7: Gravity Model - Candidate Location Selection")
    print("="*70)
    
    # POI data path
    poi_data_path = f"/workspace/China_Journal/raw_data/{city}/{city}_grid_poi.csv"
    
    if not os.path.exists(poi_data_path):
        print(f"✗ POI data not found at {poi_data_path}")
        return None
    
    try:
        # Initialize Gravity Model
        gravity_model = GravityModel(
            poi_data_path=poi_data_path,
            city=city,
            weight=1.0,
            gravity_top_n_candidates=5,
            radius=10
        )
        
        print("✓ Gravity Model initialized successfully")
        gravity_model.print_statistics()
        
        # Test with a sample current location
        current_location = 800  # Middle of the grid (approximately)
        
        print(f"\nTesting with current location: Grid {current_location}")
        
        # Get candidates for each POI category
        candidates_by_category = gravity_model.get_candidate_locations(
            current_grid_id=current_location,
            return_scores=True
        )
        
        # Print candidates for a few categories
        print(f"\nCandidates by POI Category (each category has top-{gravity_model.gravity_top_n_candidates} candidates):")
        sample_categories = ['Dining & Cusine_count', 'Shopping & Consumer Goods_count', 'Transportation Facilities_count']
        
        for category in sample_categories:
            if category in candidates_by_category:
                category_display = category.replace('_count', '')
                print(f"\n{category_display}:")
                for i, (grid_id, score) in enumerate(candidates_by_category[category][:3], 1):
                    print(f"  {i}. Grid {grid_id} - Score: {score:.4f}")
        
        print(f"\n✓ Generated {len(candidates_by_category)} POI category groups")
        print(f"  Each category has top-{gravity_model.gravity_top_n_candidates} candidates")
        print(f"  Total POI categories: {len(candidates_by_category)}")
        
        # Get top-k overall candidates
        print(f"\nTop-10 Overall Candidates (by average score):")
        top_k_candidates = gravity_model.get_top_k_candidates(
            current_grid_id=current_location,
            k=10
        )
        
        for i, (grid_id, avg_score) in enumerate(top_k_candidates, 1):
            row, col = gravity_model._grid_id_to_coordinates(grid_id)
            print(f"  {i}. Grid {grid_id} (row={row}, col={col}) - Avg Score: {avg_score:.4f}")
        
        # Get all unique candidates
        all_candidates = gravity_model.get_all_candidate_locations(
            current_grid_id=current_location,
            deduplicate=True
        )
        
        print(f"\n✓ Gravity Model test successful")
        print(f"Total unique candidates across all POI categories: {len(all_candidates)}")
        
        return gravity_model
        
    except Exception as e:
        print(f"✗ Gravity Model test failed: {e}")
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
    
    # Test 7: Gravity Model
    gravity_model = test_gravity_model(city=city)
    
    # Test 8: Integrated Prediction (LLM + RAG + Gravity)
    if rag_summary and gravity_model and rag.database_embeddings is not None:
        test_integrated_prediction(llm, rag, gravity_model, city=city)
    
    # Test 9: MobilityPredictor (Complete Pipeline)
    test_mobility_predictor(city=city)
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")


def test_integrated_prediction(llm: MobilityLLM, rag: MobilityRAG, gravity_model: GravityModel, city: str = 'beijing'):
    """Test integrated prediction combining LLM+RAG and Gravity Model."""
    print("\n" + "="*70)
    print("TEST 8: Integrated Prediction (LLM + RAG + Gravity)")
    print("="*70)
    
    # Sample query trajectory
    query_trajectory = [
        {'location_id': 780, 'timestamp': '20220706 08:00'},
        {'location_id': 781, 'timestamp': '20220706 08:30'},
        {'location_id': 800, 'timestamp': '20220706 09:00'},
    ]
    
    current_location = query_trajectory[-1]['location_id']
    
    print(f"\nQuery Trajectory:")
    for i, step in enumerate(query_trajectory, 1):
        print(f"  Step {i}: Grid {step['location_id']} at {step['timestamp']}")
    
    try:
        # Step 1: Get RAG summary
        print(f"\nStep 1: Retrieving similar trajectories from RAG database...")
        rag_summary, similar_samples, similarities = rag.generate_rag_summary(
            query_trajectory=query_trajectory,
            rag_top_m_samples=3
        )
        
        print(f"✓ Retrieved {len(similar_samples)} similar trajectories")
        print(f"  Top similarity score: {similarities[0]:.4f}")
        
        # Step 2: Get candidate locations from Gravity Model
        print(f"\nStep 2: Computing candidate locations using Gravity Model...")
        print(f"  Current location: Grid {current_location}")
        
        # Get top-k candidates from gravity model
        gravity_candidates = gravity_model.get_top_k_candidates(
            current_grid_id=current_location,
            k=20
        )
        
        candidate_locations = [grid_id for grid_id, score in gravity_candidates]
        
        print(f"✓ Generated {len(candidate_locations)} candidate locations")
        print(f"  Top-5 candidates: {candidate_locations[:5]}")
        
        # Step 3: Final prediction combining RAG summary and gravity candidates
        print(f"\nStep 3: Final prediction combining RAG summary and candidates...")
        
        predictions = llm.predict_next_location(
            query_trajectory=query_trajectory,
            candidate_locations=candidate_locations,
            rag_summary=rag_summary,
            city=city,
            top_k_predictions=5
        )
        
        print(f"\n✓ Integrated prediction successful!")
        print(f"\nFinal Top-5 Predictions:")
        for i, (loc_id, confidence) in enumerate(predictions, 1):
            row, col = gravity_model._grid_id_to_coordinates(loc_id)
            # Find if this was in gravity top candidates
            gravity_rank = next((idx for idx, (gid, _) in enumerate(gravity_candidates, 1) if gid == loc_id), None)
            gravity_info = f"(Gravity rank: {gravity_rank})" if gravity_rank else "(Not in gravity top-20)"
            print(f"  {i}. Grid {loc_id} (row={row}, col={col}) - Confidence: {confidence:.4f} {gravity_info}")
        
        return predictions
        
    except Exception as e:
        print(f"✗ Integrated prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mobility_predictor(city: str = 'beijing'):
    """Test the complete MobilityPredictor pipeline."""
    print("\n" + "="*70)
    print("TEST 9: MobilityPredictor (Complete Pipeline)")
    print("="*70)
    
    try:
        # Initialize MobilityPredictor with all modules
        predictor = MobilityPredictor(
            llm_model_name="Deepseek-R1-Distill-Qwen-3B",
            llm_model_path="/datadisk",
            rag_database_path="/workspace/China_Journal/util/rag_database",
            city=city,
            top_k_predictions=5,
            rag_top_m_samples=3,
            gravity_top_n_candidates=5,
            gravity_weight=1.0,
            gravity_radius=10,
            use_quantization=True,
            verbose=True
        )
        
        print("\n✓ MobilityPredictor initialized successfully")
        
        # Print statistics
        predictor.print_statistics()
        
        # Test prediction with sample trajectory
        print("\nTesting prediction with sample trajectory...")
        
        sample_trajectory = [
            {'location_id': 780, 'timestamp': '20220706 08:00'},
            {'location_id': 781, 'timestamp': '20220706 08:30'},
            {'location_id': 800, 'timestamp': '20220706 09:00'},
        ]
        
        ground_truth = 802  # Sample ground truth
        
        predictions, results = predictor.predict(
            observation_trajectory=sample_trajectory,
            ground_truth=ground_truth,
            print_prompt=True
        )
        
        print("\n✓ Prediction completed successfully")
        
        return predictor
        
    except Exception as e:
        print(f"✗ MobilityPredictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
