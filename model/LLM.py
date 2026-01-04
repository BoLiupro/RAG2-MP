"""
LLM Module for Mobility Prediction
This module handles LLM initialization, trajectory encoding, and summary generation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from util.utils import format_trajectory_with_distances, get_mobility_mode, calculate_grid_distance


class MobilityLLM:
    """
    Large Language Model for mobility trajectory encoding and prediction.
    This single LLM is used for both RAG encoding and final prediction.
    """
    
    def __init__(
        self,
        model_name: str = "Deepseek-R1-Distill-Qwen-3B",
        model_path: str = "/datadisk",
        use_quantization: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Mobility LLM.
        
        Args:
            model_name: Name of the LLM backbone (e.g., 'Deepseek-R1-Distill-Qwen-3B')
            model_path: Base path where models are stored
            use_quantization: Whether to use 4-bit quantization
            use_lora: Whether to use LoRA for fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            device: Device to use for inference
        """
        self.model_name = model_name
        self.model_path = f"{model_path}/{model_name}"
        self.device = device
        self.use_lora = use_lora
        
        # Configure quantization
        if use_quantization and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        stop_strings: List[str] = None
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            stop_strings: List of strings to stop generation (e.g., ['<think>', '\n\n'])
        
        Returns:
            Generated text (excluding the prompt)
        """
        # For DeepSeek-R1 models, add system message to suppress thinking
        if 'deepseek' in self.model_name.lower() or 'r1' in self.model_name.lower():
            # Prepend instruction to avoid reasoning tags
            if '<think>' not in prompt and 'Do NOT include' not in prompt:
                prompt = "Answer directly without showing your reasoning process. " + prompt
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if generated_text.startswith(prompt):
            result = generated_text[len(prompt):].strip()
        else:
            result = generated_text.strip()
        
        # Apply stop strings if provided
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in result:
                    result = result.split(stop_str)[0].strip()
        
        return result
    
    def _build_trajectory_prompt(
        self,
        trajectory: List[Dict[str, Any]],
        poi_data: Dict[int, Dict[str, float]] = None,
        city: str = "general",
        task: str = "encoding"
    ) -> str:
        """
        Build a prompt for trajectory encoding or summary generation.
        
        Args:
            trajectory: List of trajectory points, each with 'location_id', 'timestamp', etc.
            poi_data: Dictionary mapping location_id to POI features
            city: City name ('shenzhen', 'beijing', 'nanchang')
            task: Task type ('encoding' or 'summary')
        
        Returns:
            Formatted prompt string
        """
        # Get mobility mode description
        mobility_mode = get_mobility_mode(city)
        
        # Prompt for summary generation (used in RAG)
        prompt = f"Based on similar {mobility_mode} mobility patterns, "
        prompt += "Summarize its mobility pattern in terms of purpose, temporal rhythm, and functional transitions.\n\n"
        
        # Add trajectory information
        prompt += "Trajectory:\n"
        for i, point in enumerate(trajectory, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            
            # Parse timestamp
            try:
                if isinstance(timestamp, str):
                    dt = datetime.strptime(timestamp, '%Y%m%d %H:%M')
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                    weekday = dt.strftime('%A')
                    hour = dt.hour
                else:
                    time_str = str(timestamp)
                    weekday = "Unknown"
                    hour = 0
            except:
                time_str = str(timestamp)
                weekday = "Unknown"
                hour = 0
            
            # Get POI information if available
            poi_info = ""
            if poi_data and loc_id in poi_data:
                poi_features = poi_data[loc_id]
                # Find dominant POI types
                poi_types = []
                for poi_type, percentage in poi_features.items():
                    if percentage > 10:  # Only show significant POI types
                        poi_types.append(f"{poi_type}: {percentage:.1f}%")
                if poi_types:
                    poi_info = f" (POI: {', '.join(poi_types[:3])})"  # Top 3
            
            prompt += f"  {i}. Location {loc_id}, {time_str} ({weekday}){poi_info}\n"
        
        if task == "encoding":
            prompt += "\nProvide a brief semantic encoding of this trajectory pattern."
        
        return prompt
    
    def encode_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        poi_data: Dict[int, Dict[str, float]] = None,
        city: str = "general"
    ) -> np.ndarray:
        """
        Encode a trajectory into an embedding vector using LLM.
        
        Args:
            trajectory: List of trajectory points
            poi_data: POI features for locations
            city: City name
        
        Returns:
            Embedding vector as numpy array
        """
        # Build encoding prompt
        prompt = self._build_trajectory_prompt(
            trajectory, poi_data, city, task="encoding"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get hidden states (embeddings)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            # Use the last hidden state of the last token as embedding
            # Shape: (batch_size, seq_len, hidden_size)
            hidden_states = outputs.hidden_states[-1]
            
            # Get the embedding of the last token (mean pooling over sequence)
            # Or use last token: hidden_states[:, -1, :]
            embedding = hidden_states.mean(dim=1).squeeze()
            
            # Normalize the embedding
            embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding.cpu().numpy()
    
    def generate_similarity_summary(
        self,
        query_trajectory: List[Dict[str, Any]],
        similar_samples: List[Dict[str, Any]],
        poi_data: Dict[int, Dict[str, float]] = None,
        city: str = "general",
        max_length: int = 256,
        print_prompt: bool = False
    ) -> str:
        """
        Generate a summary based on similar trajectory samples.
        
        Args:
            query_trajectory: The query trajectory (observation)
            similar_samples: List of similar samples, each with 'trajectory' and 'next_location'
            poi_data: POI features for locations
            city: City name
            max_length: Maximum generation length
            print_prompt: If True, print the prompt sent to LLM
        
        Returns:
            Generated summary text
        """
        # Build prompt with query and similar samples
        if city == 'shenzhen':
            mobility_mode = "private car mobility"
        else:
            mobility_mode = "general mobility"
        
        prompt = f"You are analyzing {mobility_mode} patterns. "
        prompt += "Given the current trajectory and similar historical patterns, "
        prompt += "summarize where the user is likely to go next.\n\n"
        
        # Add query trajectory
        prompt += "Current Trajectory:\n"
        for i, point in enumerate(query_trajectory, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            prompt += f"  {i}. Location {loc_id} at {timestamp}\n"
        
        # Add similar samples
        prompt += f"\nSimilar Historical Patterns (top {len(similar_samples)}):\n"
        for idx, sample in enumerate(similar_samples[:5], 1):  # Limit to top 5
            sample_traj = sample.get('trajectory', [])
            next_loc = sample.get('next_location', 'unknown')
            
            # Show brief trajectory
            traj_locs = [str(p.get('location_id', '?')) for p in sample_traj[-3:]]  # Last 3 points
            prompt += f"  Pattern {idx}: [...{', '.join(traj_locs)}] → Next: Location {next_loc}\n"
        
        prompt += "\nBased on these patterns, summarize the likely next destination:"
        
        # Print prompt if requested
        # print_prompt parameter is for trainer to control logging
        # Model code does not print
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Increased to ensure complete responses
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if generated_text.startswith(prompt):
            summary = generated_text[len(prompt):].strip()
        else:
            summary = generated_text.strip()
        
        return summary
    
    def predict_next_location(
        self,
        query_trajectory: List[Dict[str, Any]],
        candidate_locations: List[int],
        rag_summary: str = None,
        poi_data: Dict[int, Dict[str, float]] = None,
        city: str = "general",
        top_k_predictions: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Predict the next location from candidate locations.
        
        Args:
            query_trajectory: The query trajectory (observation)
            candidate_locations: List of candidate location IDs from gravity model
            rag_summary: Summary from RAG module (optional)
            poi_data: POI features for locations
            city: City name
            top_k_predictions: Number of top predictions to return
        
        Returns:
            List of (location_id, confidence_score) tuples
        """
        # Build prediction prompt
        from util.utils import get_mobility_mode
        mobility_mode = get_mobility_mode(city)
        
        prompt = f"You are predicting the next location for {mobility_mode}.\n\n"
        
        # Add current trajectory
        prompt += "Current Trajectory:\n"
        for i, point in enumerate(query_trajectory, 1):
            loc_id = point.get('location_id', 'unknown')
            timestamp = point.get('timestamp', '')
            prompt += f"  {i}. Location {loc_id} at {timestamp}\n"
        
        # Add RAG summary if available
        if rag_summary:
            prompt += f"\nSimilar Pattern Analysis:\n{rag_summary}\n"
        
        # Add candidate locations
        prompt += f"\nCandidate Next Locations (from spatial analysis):\n"
        for loc_id in candidate_locations[:10]:  # Show top 10
            poi_info = ""
            if poi_data and loc_id in poi_data:
                poi_features = poi_data[loc_id]
                top_poi = max(poi_features.items(), key=lambda x: x[1])
                poi_info = f" - Primary: {top_poi[0]} ({top_poi[1]:.1f}%)"
            prompt += f"  - Location {loc_id}{poi_info}\n"
        
        prompt += f"\nRank the top {top_k_predictions} most likely next locations and explain briefly:"
        
        # Generate prediction
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Increased for complete responses
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse predictions (this is a simple heuristic, can be improved)
        # For now, return candidate locations with uniform confidence
        # TODO: Implement more sophisticated parsing of LLM output
        predictions = [(loc_id, 1.0 / (i + 1)) for i, loc_id in enumerate(candidate_locations[:top_k_predictions])]
        
        return predictions
    
    def save_model(self, save_path: str):
        """Save the model (especially useful if using LoRA)."""
        if self.use_lora:
            self.model.save_pretrained(save_path)
    
    def load_lora_weights(self, lora_path: str):
        """Load LoRA weights from a saved checkpoint."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
