"""
Data generators for training recommendation models
Supports both pointwise and pairwise training strategies
"""
import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Iterator, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)

class PointwiseGenerator:
    """
    Generates batches for pointwise learning (rating prediction)
    
    Each sample contains: (user, item, rating)
    Suitable for explicit feedback and rating prediction tasks
    """
    
    def __init__(self, interaction_matrix: csr_matrix, return_rating: bool = True,
                 num_negatives: int = 1, batch_size: int = 1024, 
                 shuffle: bool = True, device: str = 'cpu'):
        """
        Initialize pointwise generator
        
        Args:
            interaction_matrix: Sparse user-item interaction matrix
            return_rating: Whether to return actual ratings or binary feedback
            num_negatives: Number of negative samples per positive sample
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch
            device: Device to place tensors on
        """
        self.interaction_matrix = interaction_matrix
        self.return_rating = return_rating
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        
        self.num_users, self.num_items = interaction_matrix.shape
        self.positive_samples = list(zip(*interaction_matrix.nonzero()))
        
        # Prepare data
        self._prepare_samples()
        
    def _prepare_samples(self):
        """Prepare training samples with negative sampling"""
        self.samples = []
        
        # Add positive samples
        for user_idx, item_idx in self.positive_samples:
            rating = self.interaction_matrix[user_idx, item_idx]
            if self.return_rating:
                self.samples.append((user_idx, item_idx, rating))
            else:
                self.samples.append((user_idx, item_idx, 1.0))
        
        # Add negative samples
        if self.num_negatives > 0:
            for user_idx, item_idx in self.positive_samples:
                for _ in range(self.num_negatives):
                    # Sample negative item for this user
                    neg_item = self._sample_negative_item(user_idx)
                    self.samples.append((user_idx, neg_item, 0.0))
    
    def _sample_negative_item(self, user_idx: int) -> int:
        """Sample a negative item for a user"""
        user_items = set(self.interaction_matrix[user_idx].nonzero()[1])
        
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in user_items:
                return neg_item
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches"""
        if self.shuffle:
            random.shuffle(self.samples)
        
        for i in range(0, len(self.samples), self.batch_size):
            batch_samples = self.samples[i:i + self.batch_size]
            
            users = torch.LongTensor([s[0] for s in batch_samples]).to(self.device)
            items = torch.LongTensor([s[1] for s in batch_samples]).to(self.device)
            ratings = torch.FloatTensor([s[2] for s in batch_samples]).to(self.device)
            
            yield users, items, ratings
    
    def __len__(self) -> int:
        """Number of batches per epoch"""
        return (len(self.samples) + self.batch_size - 1) // self.batch_size


class PairwiseGenerator:
    """
    Generates batches for pairwise learning (ranking)
    
    Each sample contains: (user, positive_item, negative_item)
    Suitable for implicit feedback and ranking tasks (e.g., BPR)
    """
    
    def __init__(self, interaction_matrix: csr_matrix, num_negatives: int = 1,
                 num_positives_per_user: int = 1, batch_size: int = 1024,
                 shuffle: bool = True, device: str = 'cpu'):
        """
        Initialize pairwise generator
        
        Args:
            interaction_matrix: Sparse user-item interaction matrix
            num_negatives: Number of negative items per positive item
            num_positives_per_user: Number of positive items to sample per user
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch
            device: Device to place tensors on
        """
        self.interaction_matrix = interaction_matrix
        self.num_negatives = num_negatives
        self.num_positives_per_user = num_positives_per_user
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        
        self.num_users, self.num_items = interaction_matrix.shape
        
        # Create user-item mappings for efficient sampling
        self.user_items = {}
        for user_idx in range(self.num_users):
            items = interaction_matrix[user_idx].nonzero()[1]
            if len(items) > 0:
                self.user_items[user_idx] = items.tolist()
        
        self.active_users = list(self.user_items.keys())
        
    def _sample_triplets(self) -> list:
        """Sample training triplets (user, pos_item, neg_item)"""
        triplets = []
        
        for user_idx in self.active_users:
            user_items = self.user_items[user_idx]
            
            # Sample positive items for this user
            num_pos_samples = min(self.num_positives_per_user, len(user_items))
            pos_items = random.sample(user_items, num_pos_samples)
            
            for pos_item in pos_items:
                # Sample negative items for each positive item
                for _ in range(self.num_negatives):
                    neg_item = self._sample_negative_item(user_idx)
                    triplets.append((user_idx, pos_item, neg_item))
        
        return triplets
    
    def _sample_negative_item(self, user_idx: int) -> int:
        """Sample a negative item for a user"""
        user_items = set(self.user_items[user_idx])
        
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in user_items:
                return neg_item
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches"""
        triplets = self._sample_triplets()
        
        if self.shuffle:
            random.shuffle(triplets)
        
        for i in range(0, len(triplets), self.batch_size):
            batch_triplets = triplets[i:i + self.batch_size]
            
            users = torch.LongTensor([t[0] for t in batch_triplets]).to(self.device)
            pos_items = torch.LongTensor([t[1] for t in batch_triplets]).to(self.device)
            neg_items = torch.LongTensor([t[2] for t in batch_triplets]).to(self.device)
            
            yield users, pos_items, neg_items
    
    def __len__(self) -> int:
        """Number of batches per epoch"""
        total_triplets = sum(min(self.num_positives_per_user, len(items)) 
                           for items in self.user_items.values()) * self.num_negatives
        return (total_triplets + self.batch_size - 1) // self.batch_size


class DataLoader:
    """
    Unified data loader for both pointwise and pairwise training
    """
    
    def __init__(self, dataset, batch_size: int = 1024, training_type: str = 'pointwise',
                 num_negatives: int = 1, shuffle: bool = True, device: str = 'cpu'):
        """
        Initialize unified data loader
        
        Args:
            dataset: AmazonDataset instance
            batch_size: Batch size for training
            training_type: 'pointwise' or 'pairwise'
            num_negatives: Number of negative samples
            shuffle: Whether to shuffle data
            device: Device to place tensors on
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.training_type = training_type
        self.num_negatives = num_negatives
        self.shuffle = shuffle
        self.device = device
        
        if training_type == 'pointwise':
            self.generator = PointwiseGenerator(
                dataset.train_data,
                return_rating=True,
                num_negatives=num_negatives,
                batch_size=batch_size,
                shuffle=shuffle,
                device=device
            )
        elif training_type == 'pairwise':
            self.generator = PairwiseGenerator(
                dataset.train_data,
                num_negatives=num_negatives,
                batch_size=batch_size,
                shuffle=shuffle,
                device=device
            )
        else:
            raise ValueError(f"Unknown training_type: {training_type}")
    
    def __iter__(self):
        return iter(self.generator)
    
    def __len__(self):
        return len(self.generator)


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the components
    
    # Load Amazon Books dataset
    dataset = AmazonDataset(
        data_path="path/to/amazon_books.json.gz",
        dataset_name="books",
        min_interactions=5,
        test_ratio=0.2
    )
    
    print("Dataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create data loaders
    pointwise_loader = DataLoader(
        dataset, 
        batch_size=512, 
        training_type='pointwise',
        num_negatives=1
    )
    
    pairwise_loader = DataLoader(
        dataset,
        batch_size=512,
        training_type='pairwise', 
        num_negatives=1
    )
    
    print(f"\nPointwise batches per epoch: {len(pointwise_loader)}")
    print(f"Pairwise batches per epoch: {len(pairwise_loader)}")
    
    # Test one batch
    for batch in pointwise_loader:
        users, items, ratings = batch
        print(f"Pointwise batch - Users: {users.shape}, Items: {items.shape}, Ratings: {ratings.shape}")
        break
    
    for batch in pairwise_loader:
        users, pos_items, neg_items = batch
        print(f"Pairwise batch - Users: {users.shape}, Pos: {pos_items.shape}, Neg: {neg_items.shape}")
        break