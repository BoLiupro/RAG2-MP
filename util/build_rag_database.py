"""
Build RAG Database Script
Encodes training samples using LLM and builds FAISS index for RAG module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from model.LLM import MobilityLLM
from model.RAG import MobilityRAG
from dataset.dataset import load_datasets


def build_rag_database(config: dict):
    """
    Build RAG database from training samples.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("BUILDING RAG DATABASE")
    print("="*70)
    print(f"City: {config['data']['city']}")
    print(f"Output: {config['rag_build']['output_dir']}")
    print(f"Batch size: {config['rag_build']['batch_size']}")
    print("="*70 + "\n")
    
    # Load training data
    print("Loading training dataset...")
    train_dataset, _, _ = load_datasets(
        city=config['data']['city'],
        data_dir=config['data']['data_dir'],
        obs_len=config['data']['obs_len'],
        pred_len=config['data']['pred_len'],
        max_train_samples=config['data'].get('max_train_samples', None),
        max_val_samples=None,
        max_test_samples=None
    )
    
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Check if database already exists
    output_dir = os.path.join(
        config['rag_build']['output_dir'],
        config['data']['city']
    )
    
    if os.path.exists(output_dir) and not config['rag_build']['overwrite']:
        print(f"\nWarning: RAG database already exists at {output_dir}")
        print("Set 'overwrite: true' in config to rebuild.")
        
        response = input("Do you want to continue and overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = MobilityLLM(
        model_name=config['model']['llm_model_name'],
        model_path=config['model']['llm_model_path'],
        use_quantization=config['model']['use_quantization'],
        device='cuda'
    )
    
    # Load POI data
    poi_path = os.path.join(
        config['data']['data_dir'],
        config['data']['city'],
        'poi.csv'
    )
    
    poi_data = None
    if os.path.exists(poi_path):
        poi_data = train_dataset.poi_data
        print(f"Loaded POI data for {len(poi_data)} locations")
    else:
        print("Warning: POI data not found, proceeding without it")
    
    # Initialize RAG module
    print("\nInitializing RAG module...")
    rag = MobilityRAG(
        llm=llm,
        rag_database_path=config['rag_build']['output_dir'],
        rag_top_m_samples=config['model']['rag_top_m_samples'],
        city=config['data']['city'],
        verbose=False
    )
    
    # Set POI data
    rag.poi_data = poi_data
    
    # Prepare training samples for RAG database
    print("\nPreparing training samples...")
    training_samples = []
    
    for sample in tqdm(train_dataset, desc="Processing samples"):
        training_sample = {
            'trajectory': sample['observation'],
            'next_location': sample['next_location']
        }
        training_samples.append(training_sample)
    
    print(f"Prepared {len(training_samples)} samples for encoding")
    
    # Build RAG database
    print("\nBuilding RAG database (this may take a while)...")
    rag.build_rag_database(
        training_samples=training_samples,
        output_dir=output_dir,
        batch_size=config['rag_build']['batch_size']
    )
    
    # Save metadata
    metadata = {
        'city': config['data']['city'],
        'num_samples': len(training_samples),
        'obs_len': config['data']['obs_len'],
        'pred_len': config['data']['pred_len'],
        'llm_model': config['model']['llm_model_name'],
        'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Verify database
    print("\nVerifying RAG database...")
    success = rag.load_rag_database()
    
    if success:
        rag.print_statistics()
        print("\n" + "="*70)
        print("RAG DATABASE BUILD COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nDatabase location: {output_dir}")
        print(f"Total samples: {len(training_samples)}")
        print(f"FAISS index size: {rag.faiss_index.ntotal if rag.faiss_index else 0}")
        print("\nYou can now use this database for training and inference.")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("ERROR: Failed to verify RAG database!")
        print("="*70 + "\n")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(description='Build RAG Database for Mobility Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml file')
    parser.add_argument('--city', type=str, help='Override city from config')
    parser.add_argument('--batch-size', type=int, help='Override batch size from config')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing database')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.city:
        config['data']['city'] = args.city
    if args.batch_size:
        config['rag_build']['batch_size'] = args.batch_size
    if args.overwrite:
        config['rag_build']['overwrite'] = True
    
    print(f"Loaded configuration from {args.config}")
    print(f"\nConfiguration:")
    print(f"  City: {config['data']['city']}")
    print(f"  Training samples: {config['data'].get('max_train_samples', 'all')}")
    print(f"  Batch size: {config['rag_build']['batch_size']}")
    print(f"  Output dir: {config['rag_build']['output_dir']}")
    
    # Build database
    try:
        build_rag_database(config)
    except Exception as e:
        print(f"\nError building RAG database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
