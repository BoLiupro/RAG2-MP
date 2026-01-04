"""
Training Script for Mobility Prediction Model
Implements complete training pipeline with LLM, RAG, and Gravity Model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple

from model.Predictor import MobilityPredictor
from dataset.dataset import load_datasets
from util.utils import calculate_accuracy_at_k, calculate_mrr


class MobilityTrainer:
    """
    Trainer class for mobility prediction model.
    Handles training, validation, testing, and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(self.output_dir, 'logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.metrics_file = os.path.join(self.output_dir, 'logs', 'metrics.json')
        
        # Training history
        self.train_losses = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.log_every_n_steps = self.config['training'].get('log_every_n_steps', 50)
        
        print(f"\n{'='*70}")
        print("MOBILITY PREDICTION TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log file: {self.log_file}")
        print(f"{'='*70}\n")
    
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
        """Load train, validation, and test datasets."""
        self.log("Loading datasets...")
        
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(
            city=self.config['data']['city'],
            data_dir=self.config['data']['data_dir'],
            obs_len=self.config['data']['obs_len'],
            pred_len=self.config['data']['pred_len'],
            max_train_samples=self.config['data'].get('max_train_samples', None),
            max_val_samples=self.config['data'].get('max_val_samples', None),
            max_test_samples=self.config['data'].get('max_test_samples', None)
        )
        
        self.log(f"Train samples: {len(self.train_dataset)}")
        self.log(f"Validation samples: {len(self.val_dataset)}")
        self.log(f"Test samples: {len(self.test_dataset)}")
    
    def initialize_model(self):
        """Initialize the MobilityPredictor model."""
        self.log("Initializing MobilityPredictor model...")
        
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
            verbose=False  # Disable verbose during training
        )

        # LoRA微调：只训练LoRA参数和分类头，其余全部冻结
        use_lora = False
        if 'llm' in model_config:
            use_lora = model_config['llm'].get('use_lora', False)
        else:
            use_lora = model_config.get('use_lora', False)

        lora_params = []
        if use_lora:
            # 冻结所有参数
            for p in self.predictor.llm.model.parameters():
                p.requires_grad = False
            # 只解冻LoRA参数
            for n, p in self.predictor.llm.model.named_parameters():
                if 'lora' in n or 'lora_' in n:
                    p.requires_grad = True
                    lora_params.append(p)
            self.log(f"LoRA enabled: {len(lora_params)} trainable LLM params")
        else:
            # 完全冻结LLM
            for p in self.predictor.llm.model.parameters():
                p.requires_grad = False

        # 分类头始终可训练
        for p in self.predictor.classification_head.parameters():
            p.requires_grad = True

        # 优化器：只包含LoRA参数和分类头参数
        if not hasattr(self.predictor.llm, 'optimizer'):
            trainable_params = lora_params + list(self.predictor.classification_head.parameters())
            self.predictor.llm.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )

        self.log("Model initialized successfully")
    
    def compute_loss(
        self,
        observation: List[Dict[str, Any]],
        ground_truth: int,
        print_details: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute cross-entropy loss using classification head output.
        
        Args:
            observation: Observation trajectory
            ground_truth: Ground truth location ID
            print_details: Whether to print detailed information
        
        Returns:
            Tuple of (loss tensor, info dict)
        """
        # Get RAG summary and gravity candidates (no gradient)
        # Force eval mode for RAG to avoid dropout affecting retrieval
        model_was_training = self.predictor.llm.model.training
        self.predictor.llm.model.eval()
        self.predictor.classification_head.eval()
        self.predictor.classification_head.eval()
        with torch.no_grad():
            # RAG summary
            rag_summary, similar_samples, similarities = self.predictor.rag.generate_rag_summary(
                query_trajectory=observation,
                print_prompt=False
            )
            
            # Gravity candidates
            current_location = observation[-1]['location_id']
            candidates_by_category = self.predictor.gravity.get_candidate_locations(
                current_grid_id=current_location,
                return_scores=True
            )
        
        # Restore training mode for classification head training
        if model_was_training:
            self.predictor.llm.model.train()
        
        # Build prompt for final prediction
        prompt = self.predictor._build_prediction_prompt(
            observation_trajectory=observation,
            rag_summary=rag_summary,
            candidates_by_category=candidates_by_category
        )
        
        # Tokenize prompt
        llm = self.predictor.llm
        inputs = llm.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(llm.device)
        
        # Get LLM hidden states without tracking grads (LLM frozen)
        with torch.no_grad():
            outputs = llm.model(
                **inputs,
                output_hidden_states=True
            )

        # Use last hidden state of the last token as context encoding (detach to save memory)
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        context_encoding = last_hidden_state[:, -1, :].detach()  # (batch_size, hidden_size)
        
        # Convert to float32 for numerical stability in classification head
        if self.predictor.use_fp32_head:
            context_encoding = context_encoding.float()
        
        # Classification head predicts over ALL grids (1600)
        logits = self.predictor.classification_head(context_encoding)  # (batch_size, 1600)
        
        # Ground truth is the grid ID directly
        gt_tensor = torch.tensor([ground_truth], dtype=torch.long).to(llm.device)
        
        # Compute cross-entropy loss over all grids
        loss = F.cross_entropy(logits, gt_tensor)
        
        # Check for gradient issues
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Loss is NaN or Inf: {loss.item()}")
        
        # Get predicted probabilities for logging
        probs = F.softmax(logits, dim=1).squeeze(0)  # (1600,)
        predicted_location = torch.argmax(probs).item()
        gt_prob = probs[ground_truth].item()
        
        info = {
            'loss': loss.item(),
            'ground_truth': ground_truth,
            'predicted_location': predicted_location,
            'gt_probability': gt_prob,
            'num_grids': self.predictor.num_grids,
            'logits_max': logits.max().item(),
            'logits_min': logits.min().item(),
            'logits_mean': logits.mean().item()
        }
        
        return loss, info
    
    def evaluate_predictions(
        self,
        predictions: List[Tuple[int, float]],
        ground_truth: int
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of (location_id, probability) tuples (top-k)
            ground_truth: Ground truth location ID
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract location IDs from predictions
        predicted_locations = [loc_id for loc_id, prob in predictions]
        
        metrics = {}
        
        # Top-K accuracy
        for k in [1, 3, 5, 10]:
            if len(predicted_locations) >= k:
                metrics[f'acc@{k}'] = 1.0 if ground_truth in predicted_locations[:k] else 0.0
            else:
                metrics[f'acc@{k}'] = 1.0 if ground_truth in predicted_locations else 0.0
        
        # MRR (Mean Reciprocal Rank)
        if ground_truth in predicted_locations:
            rank = predicted_locations.index(ground_truth) + 1
            metrics['mrr'] = 1.0 / rank
        else:
            metrics['mrr'] = 0.0
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.predictor.llm.model.eval()  # LLM frozen
        self.predictor.classification_head.train()
        
        epoch_losses = []
        num_samples = min(
            len(self.train_dataset),
            self.config['training'].get('samples_per_epoch', len(self.train_dataset))
        )
        
        # Get batch size from config
        # Note: Currently processing one sample at a time due to LLM memory constraints
        # Use gradient_accumulation_steps in config to simulate larger effective batch size
        batch_size = self.config['training'].get('batch_size', 1)
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Random sample indices
        sample_indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
        
        pbar = tqdm(sample_indices, desc=f"Epoch {epoch}")
        
        for step_idx, idx in enumerate(pbar, 1):
            sample = self.train_dataset[idx]
            observation = sample['observation']
            ground_truth = sample['next_location']
            
            try:
                # Compute loss
                loss, info = self.compute_loss(observation, ground_truth,print_details=True)
                
                # Backward pass
                self.predictor.llm.optimizer.zero_grad()
                loss.backward()
                
                # Check for gradient anomalies
                total_norm = 0.0
                has_nan_grad = False
                has_inf_grad = False
                for p in self.predictor.classification_head.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        if torch.isnan(p.grad).any():
                            has_nan_grad = True
                        if torch.isinf(p.grad).any():
                            has_inf_grad = True
                total_norm = total_norm ** 0.5
                
                if has_nan_grad:
                    self.log(f"WARNING: NaN gradient detected at step {step_idx}")
                    continue
                if has_inf_grad:
                    self.log(f"WARNING: Inf gradient detected at step {step_idx}")
                    continue
                if total_norm > 100.0:
                    self.log(f"WARNING: Large gradient norm {total_norm:.2f} at step {step_idx}")
                
                # Gradient clipping (classification head only)
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.classification_head.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.predictor.llm.optimizer.step()
                
                # Log gradient statistics periodically
                if step_idx % self.log_every_n_steps == 0:
                    self.log(f"  Gradient norm: {total_norm:.4f}")
                
                epoch_losses.append(loss.item())
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Periodic logging focused on training status (no prompts or summaries)
                if step_idx % self.log_every_n_steps == 0:
                    running_avg = np.mean(epoch_losses) if epoch_losses else 0.0
                    self.log(
                        f"Epoch {epoch} | Step {step_idx}/{num_samples} | "
                        f"Batch Loss: {loss.item():.4f} | Running Avg Loss: {running_avg:.4f}",
                        print_to_console=True
                    )
                
            except Exception as e:
                self.log(f"Error processing sample {idx}: {e}", print_to_console=True)
                continue
        
        metrics = {
            'epoch': epoch,
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'train_samples': len(epoch_losses)
        }
        
        return metrics
    
    def validate(self, epoch: int = None) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number (for logging)
        
        Returns:
            Dictionary of validation metrics
        """
        self.predictor.llm.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        val_losses = []
        
        num_samples = min(
            len(self.val_dataset),
            self.config['validation'].get('num_samples', len(self.val_dataset))
        )
        
        # Get batch size from config (currently processing one at a time)
        batch_size = self.config['validation'].get('batch_size', 1)
        
        # Random sample indices
        sample_indices = np.random.choice(len(self.val_dataset), num_samples, replace=False)
        
        pbar = tqdm(sample_indices, desc="Validation")
        
        with torch.no_grad():
            for idx in pbar:
                sample = self.val_dataset[idx]
                observation = sample['observation']
                ground_truth = sample['next_location']
                
                # try:
                # Compute loss
                loss, info = self.compute_loss(observation, ground_truth)
                val_losses.append(loss.item())
                
                # Get predictions
                predictions, results = self.predictor.predict(
                    observation_trajectory=observation,
                    ground_truth=ground_truth,
                    print_prompt=False
                )
                
                all_predictions.append(predictions)
                all_ground_truths.append(ground_truth)
                    
                # except Exception as e:
                #     self.log(f"Error validating sample {idx}: {e}", print_to_console=False)
                #     continue
        
        # Compute metrics
        metrics = {
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
            'val_samples': len(all_predictions)
        }
        
        # Compute accuracy metrics
        for k in [1, 3, 5, 10]:
            acc = calculate_accuracy_at_k([all_predictions], [all_ground_truths], k)
            metrics[f'val_acc@{k}'] = acc
        
        # Compute MRR
        mrr = calculate_mrr([all_predictions], [all_ground_truths])
        metrics['val_mrr'] = mrr
        
        if epoch is not None:
            metrics['epoch'] = epoch
        
        return metrics
    
    def test(self, detailed_log: bool = True) -> Dict[str, float]:
        """
        Test the model on test set with detailed logging.
        
        Args:
            detailed_log: Whether to log detailed intermediate results
        
        Returns:
            Dictionary of test metrics
        """
        self.log("\n" + "="*70)
        self.log("TESTING MODEL")
        self.log("="*70)
        
        self.predictor.llm.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        test_losses = []
        
        num_samples = min(
            len(self.test_dataset),
            self.config['testing'].get('num_samples', len(self.test_dataset))
        )
        
        # Get batch size from config (currently processing one at a time)
        batch_size = self.config['testing'].get('batch_size', 1)
        
        # Detailed log file
        if detailed_log:
            detailed_log_file = os.path.join(
                self.output_dir,
                'logs',
                f'test_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        
        sample_indices = list(range(num_samples))
        pbar = tqdm(sample_indices, desc="Testing")
        
        with torch.no_grad():
            for sample_idx, idx in enumerate(pbar):
                sample = self.test_dataset[idx]
                observation = sample['observation']
                ground_truth = sample['next_location']
                
                try:
                    # Get predictions with intermediate results
                    predictions, results = self.predictor.predict(
                        observation_trajectory=observation,
                        ground_truth=ground_truth,
                        print_prompt=False
                    )
                    
                    # Compute loss
                    loss, info = self.compute_loss(observation, ground_truth)
                    test_losses.append(loss.item())
                    
                    all_predictions.append(predictions)
                    all_ground_truths.append(ground_truth)
                    
                    # Optional lightweight per-sample logging without prompts or summaries
                    if detailed_log and sample_idx < 10:  # Log first 10 samples only
                        with open(detailed_log_file, 'a') as f:
                            f.write(f"\n{'='*70}\n")
                            f.write(f"TEST SAMPLE {sample_idx + 1}\n")
                            f.write(f"{'='*70}\n")
                            f.write(f"Ground Truth: Grid {ground_truth}\n")
                            f.write(f"Top Predictions: {predictions}\n")
                            f.write(f"Loss: {loss.item():.4f}\n")
                            f.write(f"{'='*70}\n")
                    
                except Exception as e:
                    self.log(f"Error testing sample {idx}: {e}")
                    continue
        
        # Compute metrics
        metrics = {
            'test_loss': np.mean(test_losses) if test_losses else 0.0,
            'test_samples': len(all_predictions)
        }
        
        # Compute accuracy metrics
        for k in [1, 3, 5, 10]:
            acc = calculate_accuracy_at_k([all_predictions], [all_ground_truths], k)
            metrics[f'test_acc@{k}'] = acc
        
        # Compute MRR
        mrr = calculate_mrr([all_predictions], [all_ground_truths])
        metrics['test_mrr'] = mrr
        
        # Log metrics
        self.log("\nTest Results:")
        self.log(f"  Loss: {metrics['test_loss']:.4f}")
        self.log(f"  Acc@1: {metrics['test_acc@1']:.4f}")
        self.log(f"  Acc@3: {metrics['test_acc@3']:.4f}")
        self.log(f"  Acc@5: {metrics['test_acc@5']:.4f}")
        self.log(f"  Acc@10: {metrics['test_acc@10']:.4f}")
        self.log(f"  MRR: {metrics['test_mrr']:.4f}")
        self.log("="*70)
        
        if detailed_log:
            self.log(f"Detailed test log saved to: {detailed_log_file}")
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Metrics dictionary
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        
        # Save model state
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.predictor.llm.model.state_dict(),
            'classification_head_state_dict': self.predictor.classification_head.state_dict(),
            'optimizer_state_dict': self.predictor.llm.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)
        
        self.log(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.predictor.llm.model.state_dict(),
                'classification_head_state_dict': self.predictor.classification_head.state_dict(),
                'optimizer_state_dict': self.predictor.llm.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }, best_path)
            self.log(f"Best model saved: {best_path}")
    
    def train(self):
        """Main training loop."""
        self.log("\n" + "="*70)
        self.log("STARTING TRAINING")
        self.log("="*70)
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            self.log(f"\nEpoch {epoch}/{num_epochs}")
            self.log("-" * 70)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['train_loss'])
            
            self.log(
                f"[Epoch {epoch}] Train Samples: {train_metrics['train_samples']} | "
                f"Train Loss: {train_metrics['train_loss']:.4f}"
            )
            
            # Validate
            if epoch % self.config['training']['val_every_n_epochs'] == 0:
                val_metrics = self.validate(epoch)
                self.val_metrics.append(val_metrics)
                
                self.log(
                    f"[Epoch {epoch}] Val Samples: {val_metrics['val_samples']} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Acc@1: {val_metrics['val_acc@1']:.4f} | "
                    f"Acc@3: {val_metrics['val_acc@3']:.4f} | "
                    f"Acc@5: {val_metrics['val_acc@5']:.4f} | "
                    f"MRR: {val_metrics['val_mrr']:.4f}"
                )
                
                # Check if best model
                is_best = val_metrics['val_acc@5'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['val_acc@5']
                    self.log(f"New best model! Acc@5 improved to {self.best_val_acc:.4f}")
                
                # Save checkpoint
                if epoch % self.config['training']['save_every_n_epochs'] == 0:
                    combined_metrics = {**train_metrics, **val_metrics}
                    self.save_checkpoint(epoch, combined_metrics, is_best)
        
        # Save final metrics
        with open(self.metrics_file, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_metrics': self.val_metrics,
                'best_val_acc': self.best_val_acc
            }, f, indent=2)
        
        self.log("\n" + "="*70)
        self.log("TRAINING COMPLETED")
        self.log("="*70)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Mobility Prediction Model')
    parser.add_argument('--config', default="/workspace/China_Journal/config/config.yaml", type=str, required=False, help='Path to config.yaml file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from {args.config}")
    print(json.dumps(config, indent=2))
    
    # Initialize trainer
    trainer = MobilityTrainer(config)
    
    # Load data
    trainer.load_data()
    
    # Initialize model
    trainer.initialize_model()
    
    # Train
    trainer.train()
    
    # Test
    test_metrics = trainer.test(detailed_log=True)
    
    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
