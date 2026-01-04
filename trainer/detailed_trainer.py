"""
Detailed Training Script for Mobility Prediction Model (Debugging Mode)
This script provides verbose logging and intermediate result printing for debugging.
Uses a small subset of data for quick testing and analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple

from model.Predictor import MobilityPredictor
from dataset.dataset import load_datasets
from util.utils import calculate_accuracy_at_k, calculate_mrr, format_trajectory_with_distances


class DetailedMobilityTrainer:
    """
    Detailed trainer class for mobility prediction model with verbose debugging.
    Prints all intermediate results including prompts, RAG summaries, gravity candidates, and LLM responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detailed trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, 'logs', f'detailed_debug_{timestamp}.log')
        
        # Get debug configuration
        self.debug_config = config.get('detailed_debug', {})
        self.num_debug_samples = self.debug_config.get('num_samples', 10)
        self.print_prompts = self.debug_config.get('print_prompts', True)
        self.print_rag_results = self.debug_config.get('print_rag_results', True)
        self.print_gravity_candidates = self.debug_config.get('print_gravity_candidates', True)
        self.print_llm_responses = self.debug_config.get('print_llm_responses', True)
        
        print(f"\n{'='*80}")
        print("DETAILED MOBILITY PREDICTION TRAINER (DEBUG MODE)")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log file: {self.log_file}")
        print(f"Debug samples: {self.num_debug_samples}")
        print(f"{'='*80}\n")
    
    def log(self, message: str, print_to_console: bool = True):
        """
        Log a message to file and optionally print to console.
        
        Args:
            message: Message to log
            print_to_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        if print_to_console:
            print(message)
    
    def load_data(self):
        """Load a small subset of data for debugging."""
        self.log("Loading datasets...")
        
        # Use small subsets for quick debugging
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(
            city=self.config['data']['city'],
            data_dir=self.config['data']['data_dir'],
            obs_len=self.config['data']['obs_len'],
            pred_len=self.config['data']['pred_len'],
            max_train_samples=self.num_debug_samples,
            max_val_samples=self.num_debug_samples,
            max_test_samples=self.num_debug_samples
        )
        
        self.log(f"Train samples: {len(self.train_dataset)}")
        self.log(f"Validation samples: {len(self.val_dataset)}")
        self.log(f"Test samples: {len(self.test_dataset)}")
    
    def initialize_model(self):
        """Initialize the MobilityPredictor model with verbose mode enabled."""
        self.log("\nInitializing MobilityPredictor model...")
        
        # Support both old and new config formats for backward compatibility
        model_config = self.config['model']
        if 'llm' in model_config:  # New format
            llm_model_name = model_config['llm']['model_name']
            llm_model_path = model_config['llm']['model_path']
            use_quantization = model_config['llm']['use_quantization']
            rag_top_m_samples = model_config['rag']['top_m_samples']
            gravity_top_n_candidates = model_config['gravity']['top_n_candidates']
            gravity_weight = model_config['gravity']['weight']
            gravity_radius = model_config['gravity']['radius']
            top_k_predictions = model_config['prediction']['top_k_predictions']
        else:  # Old format
            llm_model_name = model_config['llm_model_name']
            llm_model_path = model_config['llm_model_path']
            use_quantization = model_config['use_quantization']
            rag_top_m_samples = model_config['rag_top_m_samples']
            gravity_top_n_candidates = model_config['gravity_top_n_candidates']
            gravity_weight = model_config['gravity_weight']
            gravity_radius = model_config['gravity_radius']
            top_k_predictions = model_config['top_k_predictions']
        
        self.log(f"  LLM Model: {llm_model_name}")
        self.log(f"  RAG Top-M Samples: {rag_top_m_samples}")
        self.log(f"  Gravity Top-N Candidates: {gravity_top_n_candidates}")
        self.log(f"  Prediction Top-K: {top_k_predictions}")
        
        self.predictor = MobilityPredictor(
            llm_model_name=llm_model_name,
            llm_model_path=llm_model_path,
            city=self.config['data']['city'],
            top_k_predictions=top_k_predictions,
            rag_top_m_samples=rag_top_m_samples,
            gravity_top_n_candidates=gravity_top_n_candidates,
            gravity_weight=gravity_weight,
            gravity_radius=gravity_radius,
            use_quantization=use_quantization,
            verbose=True  # Enable verbose for debugging
        )
        
        self.log("Model initialized successfully\n")
    
    def print_section_header(self, title: str):
        """Print a formatted section header."""
        self.log(f"\n{'='*80}")
        self.log(title)
        self.log(f"{'='*80}")
    
    def print_subsection_header(self, title: str):
        """Print a formatted subsection header."""
        self.log(f"\n{'-'*80}")
        self.log(title)
        self.log(f"{'-'*80}")
    
    def debug_single_prediction(self, sample_idx: int, sample: Dict[str, Any]):
        """
        Run a single prediction with detailed debugging output.
        
        Args:
            sample_idx: Sample index for logging
            sample: Sample data dictionary
        """
        self.print_section_header(f"DEBUGGING SAMPLE {sample_idx + 1}")
        
        observation = sample['observation']
        ground_truth = sample['next_location']
        
        # Print observation trajectory
        self.log(f"\n📍 Observation Trajectory ({len(observation)} points):")
        for i, point in enumerate(observation, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            self.log(f"  {i}. Location {loc_id} at {timestamp}")
        
        current_location = observation[-1]['location_id']
        self.log(f"\n📍 Current Location: Grid {current_location}")
        self.log(f"🎯 Ground Truth Next Location: Grid {ground_truth}")
        
        # Stage 1: RAG Retrieval
        self.print_subsection_header("STAGE 1: RAG RETRIEVAL")
        
        rag_summary, similar_samples, similarities = self.predictor.rag.generate_rag_summary(
            query_trajectory=observation,
            rag_top_m_samples=self.predictor.rag_top_m_samples,
            print_prompt=self.print_prompts
        )
        
        if self.print_rag_results:
            self.log(f"\n✅ Retrieved {len(similar_samples)} similar samples")
            self.log(f"\nSimilar Samples:")
            for i, (sim_sample, sim_score) in enumerate(zip(similar_samples, similarities), 1):
                self.log(f"\n  Sample {i} (Similarity: {sim_score:.4f}):")
                # Handle both 'trajectory' and 'observation' keys for compatibility
                obs = sim_sample.get('trajectory', sim_sample.get('observation', []))
                next_loc = sim_sample.get('next_location', 'unknown')
                if obs:
                    self.log(f"    Trajectory: {[p['location_id'] for p in obs]} → {next_loc}")
                else:
                    self.log(f"    Next location: {next_loc}")
            
            self.log(f"\n📄 RAG Summary:")
            self.log(f"{'-'*80}")
            self.log(rag_summary)
            self.log(f"{'-'*80}")
        
        # Stage 2: Gravity Model Candidates
        self.print_subsection_header("STAGE 2: GRAVITY MODEL CANDIDATES")
        
        candidates_by_category = self.predictor.gravity.get_candidate_locations(
            current_grid_id=current_location,
            return_scores=True,
            include_current=True
        )
        
        if self.print_gravity_candidates:
            self.log(f"\n✅ Generated candidates for {len(candidates_by_category)} POI categories")
            self.log(f"\nCategories with Candidates:")
            for i, (category, candidates) in enumerate(list(candidates_by_category.items()), 1):
                category_display = category.replace('_count', '')
                self.log(f"\n  {i}. {category_display}:")
                for j, (grid_id, score) in enumerate(candidates[:3], 1):
                    self.log(f"      {j}. Grid {grid_id} (Score: {score:.4f})")
        
        # Stage 3: LLM Final Prediction
        self.print_subsection_header("STAGE 3: LLM FINAL PREDICTION")
        
        # Build prediction prompt
        prompt = self.predictor._build_prediction_prompt(
            observation_trajectory=observation,
            rag_summary=rag_summary,
            candidates_by_category=candidates_by_category
        )
        
        if self.print_prompts:
            self.log(f"\n📝 PROMPT SENT TO LLM:")
            self.log(f"{'-'*80}")
            self.log(prompt)
            self.log(f"{'-'*80}")
        
        # Get predictions
        predictions, logits, candidate_list = self.predictor.predict(
            observation_trajectory=observation,
            ground_truth=ground_truth,
            use_beam_search=True,
            print_prompt=False  # Already printed above
        )
        
        # Print predictions
        self.log(f"\n🎲 LLM Predictions (Top-{self.predictor.top_k_predictions}):")
        predicted_locations = [loc_id for loc_id, _ in predictions]
        
        for i, (loc_id, confidence) in enumerate(predictions, 1):
            match_indicator = "✅" if loc_id == ground_truth else "  "
            self.log(f"  {match_indicator} {i}. Grid {loc_id} - Confidence: {confidence:.4f}")
        
        # Print evaluation
        if ground_truth in predicted_locations:
            rank = predicted_locations.index(ground_truth) + 1
            self.log(f"\n✅ SUCCESS: Ground truth found at rank {rank}")
        else:
            self.log(f"\n❌ MISS: Ground truth NOT in top-{self.predictor.top_k_predictions} predictions")
        
        # Calculate metrics for this sample
        metrics = {}
        for k in [1, 3, 5, 10]:
            if len(predicted_locations) >= k:
                metrics[f'acc@{k}'] = 1.0 if ground_truth in predicted_locations[:k] else 0.0
            else:
                metrics[f'acc@{k}'] = 1.0 if ground_truth in predicted_locations else 0.0
        
        if ground_truth in predicted_locations:
            rank = predicted_locations.index(ground_truth) + 1
            metrics['mrr'] = 1.0 / rank
        else:
            metrics['mrr'] = 0.0
        
        self.log(f"\n📊 Sample Metrics:")
        self.log(f"  Acc@1: {metrics['acc@1']:.2f}")
        self.log(f"  Acc@3: {metrics['acc@3']:.2f}")
        self.log(f"  Acc@5: {metrics['acc@5']:.2f}")
        self.log(f"  MRR: {metrics['mrr']:.4f}")
        
        return predictions, metrics
    
    def run_detailed_test(self):
        """Run detailed testing on a small subset of samples."""
        self.print_section_header("STARTING DETAILED TESTING (DEBUG MODE)")
        
        self.predictor.llm.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        all_metrics = []
        
        num_samples = min(len(self.test_dataset), self.num_debug_samples)
        
        self.log(f"\nTesting on {num_samples} samples...\n")
        
        with torch.no_grad():
            for idx in range(num_samples):
                sample = self.test_dataset[idx]
                
                try:
                    predictions, metrics = self.debug_single_prediction(idx, sample)
                    
                    ground_truth = sample['next_location']
                    predicted_locations = [loc_id for loc_id, _ in predictions]
                    
                    all_predictions.append(predicted_locations)
                    all_ground_truths.append(ground_truth)
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    self.log(f"\n❌ Error processing sample {idx}: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    continue
        
        # Compute overall metrics
        self.print_section_header("OVERALL TEST RESULTS")
        
        if all_metrics:
            overall_metrics = {
                'num_samples': len(all_metrics)
            }
            
            for k in [1, 3, 5, 10]:
                acc_values = [m[f'acc@{k}'] for m in all_metrics]
                overall_metrics[f'acc@{k}'] = np.mean(acc_values)
            
            mrr_values = [m['mrr'] for m in all_metrics]
            overall_metrics['mrr'] = np.mean(mrr_values)
            
            self.log(f"\n📊 Aggregate Metrics ({overall_metrics['num_samples']} samples):")
            self.log(f"  Acc@1:  {overall_metrics['acc@1']:.4f}")
            self.log(f"  Acc@3:  {overall_metrics['acc@3']:.4f}")
            self.log(f"  Acc@5:  {overall_metrics['acc@5']:.4f}")
            self.log(f"  Acc@10: {overall_metrics['acc@10']:.4f}")
            self.log(f"  MRR:    {overall_metrics['mrr']:.4f}")
            
            # Save results
            results_file = os.path.join(self.output_dir, 'logs', 'detailed_test_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'overall_metrics': overall_metrics,
                    'sample_metrics': all_metrics
                }, f, indent=2)
            
            self.log(f"\n💾 Results saved to: {results_file}")
        
        self.print_section_header("DETAILED TESTING COMPLETED")
        
        return overall_metrics if all_metrics else {}


def main():
    """Main detailed testing script."""
    parser = argparse.ArgumentParser(description='Detailed Mobility Prediction Trainer (Debug Mode)')
    parser.add_argument('--config', default="/workspace/China_Journal/config/config.yaml", type=str, required=False, help='Path to config.yaml file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from {args.config}")
    
    # Initialize trainer
    trainer = DetailedMobilityTrainer(config)
    
    # Load data
    trainer.load_data()
    
    # Initialize model
    trainer.initialize_model()
    
    # Run detailed testing
    metrics = trainer.run_detailed_test()
    
    print("\n✅ Detailed testing completed successfully!")


if __name__ == '__main__':
    main()
