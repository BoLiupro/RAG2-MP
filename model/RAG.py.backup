"""
RAG Module for Mobility Prediction
This module implements the Retrieval-Augmented Generation workflow for mobility trajectory analysis.
"""

import numpy as np
import torch
import pickle
import os
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .LLM import MobilityLLM


class MobilityRAG:
    """
    Retrieval-Augmented Generation module for mobility prediction.
    Retrieves similar historical trajectories and generates summaries.
    """
    
    def __init__(
        self,
        llm: MobilityLLM,
        rag_database_path: str = "/workspace/China_Journal/model/rag_database",
        top_m: int = 5,
        city: str = "general"
    ):
        """
        Initialize the Mobility RAG module.
        
        Args:
            llm: An initialized MobilityLLM instance
            rag_database_path: Path to the RAG database directory
            top_m: Number of top similar samples to retrieve
            city: City name for the dataset
        """
        self.llm = llm
        self.rag_database_path = rag_database_path
        self.top_m = top_m
        self.city = city
        
        # Placeholders for RAG database
        self.database_embeddings = None
        self.database_samples = None
        self.poi_data = None
        
        print(f"Initialized MobilityRAG for {city}")
        print(f"RAG database path: {rag_database_path}")
        print(f"Retrieving top-{top_m} similar samples")
    
    def load_rag_database(self, embedding_file: str = "embeddings.npy", 
                          samples_file: str = "samples.pkl"):
        """
        Load the pre-computed RAG database.
        
        Args:
            embedding_file: Filename for the embeddings array
            samples_file: Filename for the samples metadata
        """
        embedding_path = os.path.join(self.rag_database_path, self.city, embedding_file)
        samples_path = os.path.join(self.rag_database_path, self.city, samples_file)
        
        if not os.path.exists(embedding_path):
            print(f"Warning: RAG database not found at {embedding_path}")
            print("Please build the RAG database first using the trainer script.")
            return False
        
        print(f"Loading RAG database from {self.rag_database_path}...")
        
        # Load embeddings
        self.database_embeddings = np.load(embedding_path)
        print(f"Loaded {self.database_embeddings.shape[0]} embeddings")
        
        # Load samples metadata
        with open(samples_path, 'rb') as f:
            self.database_samples = pickle.load(f)
        print(f"Loaded {len(self.database_samples)} samples")
        
        return True
    
    def load_poi_data(self, poi_file: str):
        """
        Load POI data for the city.
        
        Args:
            poi_file: Path to the POI CSV file (processed grid POI data)
        """
        import pandas as pd
        
        if not os.path.exists(poi_file):
            print(f"Warning: POI file not found at {poi_file}")
            return False
        
        print(f"Loading POI data from {poi_file}...")
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
        
        print(f"Loaded POI data for {len(self.poi_data)} locations")
        return True
    
    def compute_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all database embeddings.
        
        Args:
            query_embedding: Query trajectory embedding
        
        Returns:
            Array of similarity scores
        """
        if self.database_embeddings is None:
            raise ValueError("RAG database not loaded. Call load_rag_database() first.")
        
        # Reshape query embedding if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.database_embeddings)
        
        return similarities.flatten()
    
    def retrieve_similar_samples(
        self,
        query_trajectory: List[Dict[str, Any]],
        top_m: int = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Retrieve top-m similar samples from the RAG database.
        
        Args:
            query_trajectory: The query trajectory to find similar samples for
            top_m: Number of top samples to retrieve (uses self.top_m if None)
        
        Returns:
            Tuple of (similar_samples, similarity_scores)
        """
        if top_m is None:
            top_m = self.top_m
        
        # Encode the query trajectory
        query_embedding = self.llm.encode_trajectory(
            query_trajectory,
            poi_data=self.poi_data,
            city=self.city
        )
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding)
        
        # Get top-m indices
        top_indices = np.argsort(similarities)[-top_m:][::-1]
        top_similarities = similarities[top_indices]
        
        # Retrieve corresponding samples
        similar_samples = [self.database_samples[idx] for idx in top_indices]
        
        return similar_samples, top_similarities
    
    def generate_rag_summary(
        self,
        query_trajectory: List[Dict[str, Any]],
        top_m: int = None
    ) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
        """
        Complete RAG workflow: retrieve similar samples and generate summary.
        
        Args:
            query_trajectory: The query trajectory (observation)
            top_m: Number of top samples to retrieve
        
        Returns:
            Tuple of (summary_text, similar_samples, similarity_scores)
        """
        # Retrieve similar samples
        similar_samples, similarities = self.retrieve_similar_samples(
            query_trajectory, top_m
        )
        
        print(f"\nRetrieved {len(similar_samples)} similar samples:")
        for i, (sample, sim) in enumerate(zip(similar_samples, similarities), 1):
            next_loc = sample.get('next_location', 'unknown')
            print(f"  {i}. Similarity: {sim:.4f}, Next location: {next_loc}")
        
        # Generate summary using LLM
        summary = self.llm.generate_similarity_summary(
            query_trajectory=query_trajectory,
            similar_samples=similar_samples,
            poi_data=self.poi_data,
            city=self.city
        )
        
        return summary, similar_samples, similarities
    
    def build_rag_database(
        self,
        training_samples: List[Dict[str, Any]],
        output_dir: str = None,
        batch_size: int = 32
    ):
        """
        Build the RAG database from training samples.
        This method encodes all training samples and saves embeddings.
        
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
        from tqdm import tqdm
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
            'top_m': self.top_m,
            'poi_locations': len(self.poi_data) if self.poi_data else 0
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
        print(f"Top-M retrieval: {stats['top_m']}")
        print(f"POI locations: {stats['poi_locations']}")
        print(f"{'='*50}\n")
