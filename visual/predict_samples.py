"""
Sample Prediction Script for Mobility Prediction
Reads samples from train.csv and generates predictions, saving results to CSV.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import json

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.Predictor import MobilityPredictor
from dataset.dataset import MobilityDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Predict mobility for samples from train.csv')
    parser.add_argument('--config', type=str, default='/workspace/China_Journal/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--city', type=str, default='beijing',
                        help='City name (beijing, shenzhen, nanchang). Overrides config.')
    parser.add_argument('--max_sample_num', type=int, default=10,
                        help='Maximum number of samples to predict')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file path. Default: visual/predictions_{city}.csv')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Input data file. Default: data/{city}/train.csv')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine city
    city = args.city if args.city else config['data']['city']
    
    # Determine data file path
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = os.path.join(config['data']['data_dir'], city, 'train.csv')
    
    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'predictions_{city}.csv'
    
    print(f"=" * 80)
    print(f"Mobility Prediction - Sample Testing")
    print(f"=" * 80)
    print(f"City: {city}")
    print(f"Data file: {data_file}")
    print(f"Max samples: {args.max_sample_num}")
    print(f"Output file: {output_file}")
    print(f"=" * 80)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = MobilityDataset(
        data_path=data_file,
        city=city,
        obs_len=config['data']['obs_len'],
        pred_len=config['data']['pred_len']
    )
    
    # Limit number of samples
    num_samples = min(args.max_sample_num, len(dataset))
    print(f"Will predict {num_samples} samples out of {len(dataset)} total samples")
    
    # Initialize predictor
    print("\n[2/3] Initializing predictor...")
    predictor = MobilityPredictor(
        llm_model_name=config['model']['llm']['model_name'],
        llm_model_path=config['model']['llm']['model_path'],
        rag_database_path=config['model']['rag']['database_path'],
        city=city,
        top_k_predictions=10,  # Set to 10 to get top-10 predictions
        rag_top_m_samples=config['model']['rag']['top_m_samples'],
        gravity_top_n_candidates=config['model']['gravity']['top_n_candidates'],
        gravity_weight=config['model']['gravity']['weight'],
        gravity_radius=config['model']['gravity']['radius'],
        gravity_weight_config_path=config['model']['gravity'].get('weight_config_path'),
        use_quantization=config['model']['llm']['use_quantization'],
        use_lora=False,  # Set to False for prediction only
        verbose=False,  # Don't print prompts for each sample
        temperature=config['model']['llm']['temperature'],
        top_p=config['model']['llm']['top_p'],
        top_k=config['model']['llm']['top_k'],
        do_sample=config['model']['llm']['do_sample'],
        max_new_tokens=config['model']['llm']['max_new_tokens']
    )
    
    # Make predictions
    print("\n[3/3] Making predictions...")
    results = []
    
    for idx in tqdm(range(num_samples), desc="Predicting"):
        # Get sample
        sample = dataset[idx]
        
        try:
            # Make prediction
            predictions, logits, candidate_list = predictor.predict(
                observation_trajectory=sample['observation'],
                ground_truth=sample['next_location'],
                print_prompt=False,
                use_beam_search=False
            )
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Extract prediction results
        # predictions is a list of (location_id, confidence) tuples
        top_1_pred = predictions[0][0] if len(predictions) > 0 else -1
        top_1_conf = predictions[0][1] if len(predictions) > 0 else 0.0
        
        top_10_preds = [pred[0] for pred in predictions[:10]]
        top_10_confs = [pred[1] for pred in predictions[:10]]
        
        # Pad to 10 predictions
        while len(top_10_preds) < 10:
            top_10_preds.append(-1)
            top_10_confs.append(0.0)
        
        # Calculate accuracy
        is_top1_correct = (top_1_pred == sample['next_location'])
        is_top5_correct = (sample['next_location'] in top_10_preds[:5])
        is_top10_correct = (sample['next_location'] in top_10_preds)
        
        # Prepare result
        result = {
            'sample_idx': idx,
            'user_id': sample['user_id'],
            'ground_truth': sample['next_location'],
            'top1_prediction': top_1_pred,
            'top1_confidence': top_1_conf,
            'top1_correct': is_top1_correct,
            'top5_correct': is_top5_correct,
            'top10_correct': is_top10_correct,
            # Top 10 predictions
            'pred_1': top_10_preds[0],
            'conf_1': top_10_confs[0],
            'pred_2': top_10_preds[1],
            'conf_2': top_10_confs[1],
            'pred_3': top_10_preds[2],
            'conf_3': top_10_confs[2],
            'pred_4': top_10_preds[3],
            'conf_4': top_10_confs[3],
            'pred_5': top_10_preds[4],
            'conf_5': top_10_confs[4],
            'pred_6': top_10_preds[5],
            'conf_6': top_10_confs[5],
            'pred_7': top_10_preds[6],
            'conf_7': top_10_confs[6],
            'pred_8': top_10_preds[7],
            'conf_8': top_10_confs[7],
            'pred_9': top_10_preds[8],
            'conf_9': top_10_confs[8],
            'pred_10': top_10_preds[9],
            'conf_10': top_10_confs[9],
            # Observation trajectory (last 3 locations)
            'obs_loc_-3': sample['observation'][-3]['location_id'] if len(sample['observation']) >= 3 else -1,
            'obs_loc_-2': sample['observation'][-2]['location_id'] if len(sample['observation']) >= 2 else -1,
            'obs_loc_-1': sample['observation'][-1]['location_id'],
        }
        
        results.append(result)
    
    # Save results to CSV
    print(f"\n[Done] Saving results to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total samples predicted: {len(results)}")
    
    if len(results) > 0:
        top1_acc = results_df['top1_correct'].mean() * 100
        top5_acc = results_df['top5_correct'].mean() * 100
        top10_acc = results_df['top10_correct'].mean() * 100
        avg_conf = results_df['top1_confidence'].mean()
        
        print(f"Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
        print(f"Top-10 Accuracy: {top10_acc:.2f}%")
        print(f"Average Top-1 Confidence: {avg_conf:.4f}")
        
        # Print sample results
        print(f"\nFirst 5 prediction results:")
        print(results_df[['sample_idx', 'ground_truth', 'top1_prediction', 
                         'top1_confidence', 'top1_correct', 'top5_correct', 'top10_correct']].head())
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
