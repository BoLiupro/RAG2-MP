"""
RAG Module for Mobility Prediction with FAISS
This module implements the Retrieval-Augmented Generation workflow for mobility trajectory analysis.
Uses FAISS for efficient similarity search.
"""

import numpy as np
import torch
import pickle
import os
import json
import re
import faiss
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .LLM import MobilityLLM
from util.utils import format_trajectory_with_distances, calculate_grid_distance, format_candidates_with_distances


class MobilityRAG:
    """
    Retrieval-Augmented Generation module for mobility prediction.
    Retrieves similar historical trajectories and generates summaries using FAISS.
    """
    
    def __init__(
        self,
        llm: MobilityLLM,
        rag_database_path: str = "/workspace/China_Journal/util/rag_database",
        rag_top_m_samples: int = 5,
        city: str = "general",
        verbose: bool = False
    ):
        """
        Initialize the Mobility RAG module.
        
        Args:
            llm: An initialized MobilityLLM instance
            rag_database_path: Path to the RAG database directory
            rag_top_m_samples: Number of top similar samples to retrieve
            city: City name for the dataset
            verbose: If True, print prompts sent to LLM
        """
        self.llm = llm
        self.rag_database_path = rag_database_path
        self.rag_top_m_samples = rag_top_m_samples
        self.city = city
        self.verbose = verbose
        
        # Placeholders for RAG database
        self.database_embeddings = None
        self.database_samples = None
        self.poi_data = None
        self.faiss_index = None
    
    def _build_faiss_index(self):
        """
        Build FAISS index from database embeddings.
        Uses Inner Product (IP) index since embeddings are normalized.
        For normalized vectors, inner product = cosine similarity.
        """
        if self.database_embeddings is None:
            raise ValueError("Database embeddings not loaded")
        
        dimension = self.database_embeddings.shape[1]
        
        # Use IndexFlatIP for exact search with inner product (cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(self.database_embeddings.astype('float32'))
    
    def load_rag_database(self, embedding_file: str = "embeddings.npy", 
                          samples_file: str = "samples.pkl"):
        """
        Load the pre-computed RAG database with FAISS index.
        
        Args:
            embedding_file: Filename for the embeddings array
            samples_file: Filename for the samples metadata
        """
        embedding_path = os.path.join(self.rag_database_path, self.city, embedding_file)
        samples_path = os.path.join(self.rag_database_path, self.city, samples_file)
        faiss_index_path = os.path.join(self.rag_database_path, self.city, "faiss_index.bin")
        
        if not os.path.exists(embedding_path):
            return False
        
        # Load embeddings
        self.database_embeddings = np.load(embedding_path)
        
        # Load samples metadata
        with open(samples_path, 'rb') as f:
            self.database_samples = pickle.load(f)
        
        # Load or build FAISS index
        if os.path.exists(faiss_index_path):
            self.faiss_index = faiss.read_index(faiss_index_path)
        else:
            self._build_faiss_index()
            # Save FAISS index
            faiss.write_index(self.faiss_index, faiss_index_path)
        
        return True
    
    def load_poi_data(self, poi_file: str):
        """
        Load POI data for the city.
        
        Args:
            poi_file: Path to the POI CSV file (processed grid POI data)
        """
        import pandas as pd
        
        if not os.path.exists(poi_file):
            return False
        
        poi_df = pd.read_csv(poi_file)
        
        # Convert to dictionary format: {location_id: {poi_type: percentage}}
        self.poi_data = {}
        
        # Get POI percentage columns (columns ending with '_percentage')
        poi_columns = [col for col in poi_df.columns if col.endswith('_percentage')]
        
        for _, row in poi_df.iterrows():
            grid_id = row['grid_id']
            poi_features = {}
            for col in poi_columns:
                poi_type = col.replace('_percentage', '')
                poi_features[poi_type] = row[col]
            self.poi_data[grid_id] = poi_features
        
        return True
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarity using FAISS for efficient search.
        
        Args:
            query_embedding: Query embedding (normalized)
            top_k: Number of top results to return (if None, use self.rag_top_m_samples)
        
        Returns:
            Tuple of (similarities, indices) for top-k matches
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Load RAG database first.")
        
        if top_k is None:
            top_k = self.rag_top_m_samples
        
        # Reshape query embedding if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # FAISS search (returns distances and indices)
        # For normalized vectors with IndexFlatIP, distance = cosine similarity
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        return similarities[0], indices[0]
    
    def retrieve_similar_samples(
        self,
        query_trajectory: List[Dict[str, Any]],
        rag_top_m_samples: int = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Retrieve top-m similar samples from the RAG database using FAISS.
        
        Args:
            query_trajectory: The query trajectory to find similar samples for
            rag_top_m_samples: Number of top samples to retrieve (uses self.rag_top_m_samples if None)
        
        Returns:
            Tuple of (similar_samples, similarity_scores, indices)
        """
        if rag_top_m_samples is None:
            rag_top_m_samples = self.rag_top_m_samples
        
        # Encode the query trajectory
        query_embedding = self.llm.encode_trajectory(
            query_trajectory,
            poi_data=self.poi_data,
            city=self.city
        )
        
        # Use FAISS for efficient similarity search
        similarities, indices = self.compute_similarity(query_embedding, top_k=rag_top_m_samples)
        
        # Retrieve corresponding samples
        similar_samples = [self.database_samples[idx] for idx in indices]
        
        return similar_samples, similarities, indices
    
    def generate_rag_summary(
        self,
        query_trajectory: List[Dict[str, Any]],
        rag_top_m_samples: int = None,
        print_prompt: bool = None
    ) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
        """
        Complete RAG workflow: retrieve similar samples and generate summary with structured format.
        
        Args:
            query_trajectory: The query trajectory (observation)
            rag_top_m_samples: Number of top samples to retrieve
            print_prompt: If True, print the prompt sent to LLM (overrides self.verbose)
        
        Returns:
            Tuple of (summary_text, similar_samples, similarity_scores)
        """
        # Use verbose setting if print_prompt not specified
        if print_prompt is None:
            print_prompt = self.verbose
        
        # Retrieve similar samples using FAISS
        similar_samples, similarities, indices = self.retrieve_similar_samples(
            query_trajectory, rag_top_m_samples
        )
        
        # print(f"\nRetrieved {len(similar_samples)} similar samples (using FAISS):")
        # for i, (sample, sim, idx) in enumerate(zip(similar_samples, similarities, indices), 1):
        #     next_loc = sample.get('next_location', 'unknown')
        #     print(f"  {i}. Similarity: {sim:.4f}, Next location: {next_loc}, Index: {idx}")
        
        # Generate structured summary with location-based grouping
        summary = self._generate_structured_summary(
            query_trajectory=query_trajectory,
            similar_samples=similar_samples,
            similarities=similarities,
            print_prompt=print_prompt
        )
        
        return summary, similar_samples, similarities
    
    def _generate_structured_summary(
        self,
        query_trajectory: List[Dict[str, Any]],
        similar_samples: List[Dict[str, Any]],
        similarities: np.ndarray,
        print_prompt: bool = False,
        max_summary_length: int = 500
    ) -> str:
        """
        Generate an LLM-synthesized summary from similar trajectory patterns.
        Uses LLM to extract mobility patterns and insights from retrieved samples.
        
        Args:
            query_trajectory: The query trajectory
            similar_samples: List of similar trajectory samples
            similarities: Similarity scores for each sample
            print_prompt: If True, print the prompt
            max_summary_length: Maximum length of summary in tokens (approximate)
        
        Returns:
            LLM-generated summary text with pattern analysis
        """
        from util.utils import get_mobility_mode
        
        # Group samples by next location for statistical analysis
        location_groups = {}
        current_loc = query_trajectory[-1]['location_id']
        
        for sample, sim in zip(similar_samples, similarities):
            next_loc = sample.get('next_location', None)
            if next_loc is not None:
                if next_loc not in location_groups:
                    location_groups[next_loc] = []
                location_groups[next_loc].append((sample, sim))
        
        # Calculate location statistics
        location_stats = []
        for loc, samples_list in location_groups.items():
            avg_sim = np.mean([sim for _, sim in samples_list])
            frequency = len(samples_list)
            
            # Calculate distance
            dist = calculate_grid_distance(current_loc, loc, city=self.city)
            
            # Get POI info
            poi_types = []
            if self.poi_data and loc in self.poi_data:
                poi_features = self.poi_data[loc]
                top_pois = sorted(poi_features.items(), key=lambda x: x[1], reverse=True)[:2]
                poi_types = [poi[0] for poi in top_pois if poi[1] > 10]
            
            # Collect temporal patterns
            timestamps = []
            for sample, _ in samples_list:
                traj = sample.get('trajectory', [])
                if traj:
                    ts = traj[-1].get('timestamp', '')
                    if ts:
                        timestamps.append(ts)
            
            location_stats.append({
                'grid_id': loc,
                'frequency': frequency,
                'avg_similarity': avg_sim,
                'distance_km': dist,
                'poi_types': poi_types,
                'timestamps': timestamps
            })
        
        # Sort by frequency and similarity
        location_stats.sort(key=lambda x: (x['frequency'], x['avg_similarity']), reverse=True)
        
        # Build analysis prompt for LLM
        mobility_mode = get_mobility_mode(self.city)
        
        # Prepare context for LLM
        context_lines = []
        context_lines.append(f"Query trajectory ends at Grid {current_loc}")
        context_lines.append(f"\nRetrieved {len(similar_samples)} similar {mobility_mode} patterns:")
        
        for i, stat in enumerate(location_stats[:5], 1):
            line = f"\n{i}. Grid {stat['grid_id']}: "
            line += f"{stat['frequency']} occurrences, "
            line += f"distance {stat['distance_km']:.2f} km, "
            line += f"similarity {stat['avg_similarity']:.3f}"
            
            if stat['poi_types']:
                line += f", area type: {', '.join(stat['poi_types'])}"
            
            if stat['timestamps']:
                # Analyze time patterns
                try:
                    from datetime import datetime
                    hours = []
                    weekdays = []
                    for ts in stat['timestamps'][:3]:  # Sample first 3
                        dt = datetime.strptime(ts, '%Y%m%d %H:%M')
                        hours.append(dt.hour)
                        weekdays.append(dt.strftime('%A'))
                    if hours:
                        line += f", time patterns: {hours[0]}:00-{hours[-1]}:00"
                    if weekdays:
                        line += f", days: {', '.join(set(weekdays))}"
                except:
                    pass
            
            context_lines.append(line)
        
        context = "\n".join(context_lines)
        
        # Structured synthesis prompt with JSON format requirement
        synthesis_prompt = f"""Analyze the following mobility patterns and provide a structured summary in JSON format.

{context}

You must respond with ONLY a valid JSON object in this exact format (no additional text, explanations, or markdown):

{{
  "next_locations": [
    {{
      "grid_id": <grid_id>,
      "frequency": <number>,
      "distance_km": <number>,
      "area_type": "<poi_types or 'N/A'>",
      "reason": "<concise reason why this location is likely>"
    }}
  ],
  "spatial_patterns": "<describe distance trends and area characteristics in 1-2 sentences>",
  "temporal_patterns": "<describe time patterns in 1-2 sentences, or 'No clear temporal pattern'>"
}}

Include top 3-5 next locations. Keep total response under 200 words."""
        
        if print_prompt:
            print(f"\n{'='*70}")
            print("LLM SYNTHESIS PROMPT:")
            print(f"{'='*70}")
            print(synthesis_prompt)
            print(f"{'='*70}\n")
        
        # Generate summary using LLM with JSON format
        try:
            llm_response = self.llm.generate(
                prompt=synthesis_prompt,
                max_new_tokens=min(max_summary_length, 500),  # Increased for JSON output
                temperature=0.1,  # Very low temperature for structured output
                do_sample=False  # Disable sampling for more deterministic output
            )
            
            # Clean up the response
            llm_response = llm_response.strip()
            
            if print_prompt:
                print(f"\n{'='*70}")
                print("LLM RAW RESPONSE:")
                print(f"{'='*70}")
                print(llm_response)
                print(f"{'='*70}\n")
            
            # Parse JSON response
            summary_json = self._parse_json_response(llm_response)
            
            # Convert JSON to formatted text for downstream use
            summary = self._format_json_to_text(summary_json)
            
        except Exception as e:
            print(f"Warning: LLM synthesis or JSON parsing failed: {e}")
            print("Falling back to structured summary...")
            
            # Fallback: create JSON structure directly from data
            summary_json = self._create_fallback_json(location_stats, mobility_mode, len(similar_samples))
            summary = self._format_json_to_text(summary_json)
        
        if print_prompt:
            print(f"\n{'='*70}")
            print("GENERATED SUMMARY (TEXT FORMAT):")
            print(f"{'='*70}")
            print(summary)
            print(f"{'='*70}\n")
        
        return summary
    
    def _parse_json_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM output.
        Handles various formats including markdown code blocks.
        
        Args:
            llm_response: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        llm_response = re.sub(r'```json\s*', '', llm_response)
        llm_response = re.sub(r'```\s*', '', llm_response)
        
        # Remove any leading/trailing whitespace
        llm_response = llm_response.strip()
        
        # Try to find JSON object in the response
        # Look for content between first { and last }
        start_idx = llm_response.find('{')
        end_idx = llm_response.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in LLM response")
        
        json_str = llm_response[start_idx:end_idx + 1]
        
        try:
            summary_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON errors
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            try:
                summary_json = json.loads(json_str)
            except:
                raise ValueError(f"Failed to parse JSON: {e}")
        
        # Validate required fields
        required_fields = ['next_locations', 'spatial_patterns', 'temporal_patterns']
        for field in required_fields:
            if field not in summary_json:
                raise ValueError(f"Missing required field: {field}")
        
        return summary_json
    
    def _format_json_to_text(self, summary_json: Dict[str, Any]) -> str:
        """
        Convert JSON summary to formatted text for downstream use.
        
        Args:
            summary_json: Parsed JSON dictionary
            
        Returns:
            Formatted text summary
        """
        summary_lines = []
        
        # Next locations section
        summary_lines.append("**Next Locations:**")
        next_locs = summary_json.get('next_locations', [])
        for i, loc in enumerate(next_locs, 1):
            grid_id = loc.get('grid_id', 'N/A')
            
            # Handle type conversion - LLM might return strings instead of numbers
            freq_raw = loc.get('frequency', 0)
            freq = int(freq_raw) if isinstance(freq_raw, (int, float)) else (int(freq_raw) if str(freq_raw).isdigit() else 0)
            
            dist_raw = loc.get('distance_km', 0)
            try:
                dist = float(dist_raw) if isinstance(dist_raw, (int, float, str)) else 0
            except (ValueError, TypeError):
                dist = 0.0
            
            area = loc.get('area_type', 'N/A')
            reason = loc.get('reason', 'Similar pattern match')
            
            line = f"- Grid {grid_id}: {freq} patterns, {dist:.2f} km"
            if area and area != 'N/A':
                line += f", {area}"
            line += f"\n  Reason: {reason}"
            summary_lines.append(line)
        
        # Spatial patterns section
        spatial = summary_json.get('spatial_patterns', 'No spatial pattern information')
        summary_lines.append(f"\n**Spatial Patterns:**")
        summary_lines.append(spatial)
        
        # Temporal patterns section
        temporal = summary_json.get('temporal_patterns', 'No temporal pattern information')
        summary_lines.append(f"\n**Temporal Patterns:**")
        summary_lines.append(temporal)
        
        return "\n".join(summary_lines)
    
    def _create_fallback_json(
        self,
        location_stats: List[Dict[str, Any]],
        mobility_mode: str,
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Create fallback JSON structure when LLM fails.
        
        Args:
            location_stats: Location statistics
            mobility_mode: Mode of mobility (car, phone, etc.)
            num_samples: Number of similar samples
            
        Returns:
            JSON dictionary with summary structure
        """
        # Build next locations list
        next_locations = []
        for i, stat in enumerate(location_stats[:5], 1):
            reason_parts = []
            if stat['frequency'] > 1:
                reason_parts.append(f"{stat['frequency']} historical visits")
            if stat['distance_km'] < 2:
                reason_parts.append("close proximity")
            elif stat['distance_km'] > 10:
                reason_parts.append("long-distance movement")
            if stat['poi_types']:
                reason_parts.append(f"{stat['poi_types'][0]} area")
            
            reason = ", ".join(reason_parts) if reason_parts else "similar pattern match"
            area_type = ", ".join(stat['poi_types']) if stat['poi_types'] else "N/A"
            
            next_locations.append({
                "grid_id": stat['grid_id'],
                "frequency": stat['frequency'],
                "distance_km": round(stat['distance_km'], 2),
                "area_type": area_type,
                "reason": reason
            })
        
        # Spatial patterns
        if location_stats:
            avg_dist = np.mean([s['distance_km'] for s in location_stats[:3]])
            poi_diversity = len(set([p for s in location_stats[:3] for p in s['poi_types']]))
            
            spatial_desc = f"Average distance {avg_dist:.2f} km. "
            if poi_diversity > 1:
                spatial_desc += f"Diverse area types ({poi_diversity} categories)."
            else:
                spatial_desc += "Consistent area type preference."
        else:
            spatial_desc = "Insufficient data for spatial pattern analysis."
        
        # Temporal patterns
        has_temporal = any(s['timestamps'] for s in location_stats[:3])
        if has_temporal:
            temporal_desc = f"Based on {num_samples} similar {mobility_mode} patterns, movements observed during typical commute and activity hours."
        else:
            temporal_desc = "Insufficient temporal data for pattern analysis."
        
        return {
            "next_locations": next_locations,
            "spatial_patterns": spatial_desc,
            "temporal_patterns": temporal_desc
        }
    
    def build_rag_database(
        self,
        training_samples: List[Dict[str, Any]],
        output_dir: str = None,
        batch_size: int = 32
    ):
        """
        Build the RAG database from training samples with FAISS index.
        This method encodes all training samples and saves embeddings and FAISS index.
        
        Args:
            training_samples: List of training trajectory samples
                Each sample should have 'trajectory' and 'next_location'
            output_dir: Output directory for the database (uses self.rag_database_path if None)
            batch_size: Batch size for encoding
        """
        if output_dir is None:
            output_dir = os.path.join(self.rag_database_path, self.city)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Building RAG Database for {self.city}")
        print(f"{'='*70}")
        print(f"Total samples: {len(training_samples)}")
        print(f"Output directory: {output_dir}")
        
        embeddings = []
        
        # Encode all samples
        for i in tqdm(range(0, len(training_samples), batch_size), desc="Encoding samples"):
            batch_samples = training_samples[i:i + batch_size]
            
            for sample in batch_samples:
                trajectory = sample['trajectory']
                embedding = self.llm.encode_trajectory(
                    trajectory,
                    poi_data=self.poi_data,
                    city=self.city
                )
                embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Save embeddings
        embedding_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embedding_path, embeddings)
        print(f"\nSaved embeddings to {embedding_path}")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Save samples metadata
        samples_path = os.path.join(output_dir, "samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump(training_samples, f)
        print(f"Saved samples metadata to {samples_path}")
        
        # Update internal state
        self.database_embeddings = embeddings
        self.database_samples = training_samples
        
        # Build and save FAISS index
        print(f"\nBuilding FAISS index...")
        self._build_faiss_index()
        
        faiss_index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.faiss_index, faiss_index_path)
        print(f"Saved FAISS index to {faiss_index_path}")
        
        print(f"{'='*70}")
        print("RAG database built successfully!")
        print(f"{'='*70}\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            'database_size': len(self.database_samples) if self.database_samples else 0,
            'embedding_dim': self.database_embeddings.shape[1] if self.database_embeddings is not None else 0,
            'city': self.city,
            'rag_top_m_samples': self.rag_top_m_samples,
            'poi_locations': len(self.poi_data) if self.poi_data else 0,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
        return stats
    
    def print_statistics(self):
        """Print RAG database statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*50}")
        print("RAG Database Statistics")
        print(f"{'='*50}")
        print(f"City: {stats['city']}")
        print(f"Database size: {stats['database_size']} samples")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"RAG Top-M Samples: {stats['rag_top_m_samples']}")
        print(f"POI locations: {stats['poi_locations']}")
        print(f"FAISS index size: {stats['faiss_index_size']}")
        print(f"{'='*50}\n")
