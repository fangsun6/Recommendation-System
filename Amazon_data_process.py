"""
Amazon Review Dataset Handler
Processes Amazon review data for recommendation systems
"""
import pandas as pd
import numpy as np
import torch
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import gzip
from collections import defaultdict

logger = logging.getLogger(__name__)

class AmazonDataset:
    """
    Amazon Review Dataset for recommendation systems
    
    Supports multiple Amazon datasets:
    - Books
    - Electronics  
    - Movies and TV
    - Clothing
    - etc.
    """
    
    def __init__(self, data_path: str, dataset_name: str = "books", 
                 min_interactions: int = 5, test_ratio: float = 0.2):
        """
        Initialize Amazon dataset
        
        Args:
            data_path: Path to Amazon review data file
            dataset_name: Name of the dataset (books, electronics, etc.)
            min_interactions: Minimum interactions per user/item to keep
            test_ratio: Ratio of data to use for testing
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.min_interactions = min_interactions
        self.test_ratio = test_ratio
        
        # Data containers
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # Dataset statistics
        self.num_users = 0
        self.num_items = 0
        self.num_interactions = 0
        
        # Load and process data
        self._load_data()
        self._preprocess_data()
        self._create_train_test_split()
        
        logger.info(f"Amazon {dataset_name} dataset loaded: "
                   f"{self.num_users} users, {self.num_items} items, "
                   f"{self.num_interactions} interactions")
    
    def _load_data(self):
        """Load Amazon review data from file"""
        try:
            if self.data_path.suffix == '.gz':
                # Handle gzipped JSON files
                data = []
                with gzip.open(self.data_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                self.raw_data = pd.DataFrame(data)
            elif self.data_path.suffix == '.json':
                # Handle regular JSON files
                data = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                self.raw_data = pd.DataFrame(data)
            elif self.data_path.suffix == '.csv':
                # Handle CSV files
                self.raw_data = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
                
            logger.info(f"Loaded {len(self.raw_data)} raw interactions")
            
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise
    
    def _preprocess_data(self):
        """Preprocess the raw Amazon data"""
        # Standardize column names
        column_mapping = {
            'reviewerID': 'user_id',
            'asin': 'item_id', 
            'overall': 'rating',
            'unixReviewTime': 'timestamp'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in self.raw_data.columns:
                self.raw_data = self.raw_data.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['user_id', 'item_id', 'rating']
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove duplicates
        initial_size = len(self.raw_data)
        self.raw_data = self.raw_data.drop_duplicates(subset=['user_id', 'item_id'])
        logger.info(f"Removed {initial_size - len(self.raw_data)} duplicate interactions")
        
        # Filter by minimum interactions
        self._filter_by_interactions()
        
        # Create user and item mappings
        self._create_mappings()
        
        # Convert to implicit feedback (ratings >= 4 are positive)
        self.raw_data['implicit_rating'] = (self.raw_data['rating'] >= 4.0).astype(int)
    
    def _filter_by_interactions(self):
        """Filter users and items by minimum interaction count"""
        if self.min_interactions <= 1:
            return
        
        # Iteratively filter until convergence
        prev_size = 0
        current_size = len(self.raw_data)
        
        while current_size != prev_size:
            prev_size = current_size
            
            # Filter users with insufficient interactions
            user_counts = self.raw_data['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            self.raw_data = self.raw_data[self.raw_data['user_id'].isin(valid_users)]
            
            # Filter items with insufficient interactions
            item_counts = self.raw_data['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            self.raw_data = self.raw_data[self.raw_data['item_id'].isin(valid_items)]
            
            current_size = len(self.raw_data)
        
        logger.info(f"After filtering: {len(self.raw_data)} interactions remain")
    
    def _create_mappings(self):
        """Create user and item ID mappings"""
        # Create user mapping
        unique_users = self.raw_data['user_id'].unique()
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        
        # Create item mapping
        unique_items = self.raw_data['item_id'].unique()
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # Update dataset statistics
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        self.num_interactions = len(self.raw_data)
        
        # Map IDs in dataframe
        self.raw_data['user_idx'] = self.raw_data['user_id'].map(self.user_map)
        self.raw_data['item_idx'] = self.raw_data['item_id'].map(self.item_map)
    
    def _create_train_test_split(self):
        """Create train/test split using temporal or random splitting"""
        if 'timestamp' in self.raw_data.columns:
            # Temporal split - use last interactions as test
            self.raw_data = self.raw_data.sort_values('timestamp')
            split_idx = int(len(self.raw_data) * (1 - self.test_ratio))
            train_df = self.raw_data.iloc[:split_idx]
            test_df = self.raw_data.iloc[split_idx:]
        else:
            # Random split
            train_df = self.raw_data.sample(frac=1-self.test_ratio, random_state=42)
            test_df = self.raw_data.drop(train_df.index)
        
        # Create sparse matrices
        self.train_data = self._create_sparse_matrix(train_df)
        self.test_data = self._create_sparse_matrix(test_df)
        
        logger.info(f"Train interactions: {len(train_df)}, Test interactions: {len(test_df)}")
    
    def _create_sparse_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Create sparse interaction matrix from dataframe"""
        rows = df['user_idx'].values
        cols = df['item_idx'].values
        data = df['implicit_rating'].values
        
        matrix = csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
        return matrix
    
    def get_user_items(self, user_idx: int, interaction_type: str = 'train') -> List[int]:
        """Get items interacted by a user"""
        matrix = self.train_data if interaction_type == 'train' else self.test_data
        user_items = matrix[user_idx].nonzero()[1]
        return user_items.tolist()
    
    def get_item_users(self, item_idx: int, interaction_type: str = 'train') -> List[int]:
        """Get users who interacted with an item"""
        matrix = self.train_data if interaction_type == 'train' else self.test_data
        item_users = matrix[:, item_idx].nonzero()[0]
        return item_users.tolist()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        train_interactions = self.train_data.nnz
        test_interactions = self.test_data.nnz
        sparsity = 1 - (train_interactions / (self.num_users * self.num_items))
        
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'train_interactions': train_interactions,
            'test_interactions': test_interactions,
            'total_interactions': train_interactions + test_interactions,
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'avg_interactions_per_user': train_interactions / self.num_users,
            'avg_interactions_per_item': train_interactions / self.num_items
        }
    
    def save_processed_data(self, output_dir: str):
        """Save processed dataset for future use"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save mappings
        mappings = {
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        
        with open(output_path / 'mappings.json', 'w') as f:
            # Convert numpy int64 to regular int for JSON serialization
            json_mappings = {}
            for key, mapping in mappings.items():
                if isinstance(list(mapping.values())[0], (np.integer, int)):
                    json_mappings[key] = {k: int(v) for k, v in mapping.items()}
                else:
                    json_mappings[key] = mapping
            json.dump(json_mappings, f, indent=2)
        
        # Save sparse matrices
        from scipy.sparse import save_npz
        save_npz(output_path / 'train_matrix.npz', self.train_data)
        save_npz(output_path / 'test_matrix.npz', self.test_data)
        
        # Save statistics
        stats = self.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processed data saved to {output_path}")