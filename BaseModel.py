"""
Base Model class for recommendation systems
Provides common functionality and interface for all recommendation models
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all recommendation models.
    
    Provides common functionality and enforces interface consistency
    across different recommendation algorithms.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = "BaseModel"
        self.device = torch.device("cpu")
        
    @abstractmethod
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            torch.Tensor: Predicted ratings/scores
        """
        pass
    
    @abstractmethod
    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None) -> Dict:
        """
        Train the model
        
        Args:
            dataset: Training dataset
            exp_config: Experiment configuration
            evaluator: Optional evaluator for validation
            early_stop: Optional early stopping mechanism
            loggers: Optional list of loggers
            
        Returns:
            Dict: Training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, eval_users: np.ndarray, eval_pos: np.ndarray, test_batch_size: int) -> np.ndarray:
        """
        Generate predictions for evaluation
        
        Args:
            eval_users: Array of user IDs to evaluate
            eval_pos: Positive interactions matrix
            test_batch_size: Batch size for prediction
            
        Returns:
            np.ndarray: Prediction matrix
        """
        pass
    
    def predict_batch_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for a batch of users across all items
        
        Args:
            user_ids: Tensor of user IDs
            
        Returns:
            torch.Tensor: Predictions for all items
        """
        batch_size = user_ids.size(0)
        all_items = torch.arange(self.num_items, device=self.device)
        
        # Expand user_ids to match all items
        user_ids_expanded = user_ids.unsqueeze(1).expand(batch_size, self.num_items)
        all_items_expanded = all_items.unsqueeze(0).expand(batch_size, self.num_items)
        
        # Flatten for forward pass
        user_flat = user_ids_expanded.flatten()
        items_flat = all_items_expanded.flatten()
        
        # Get predictions and reshape
        predictions = self.forward(user_flat, items_flat)
        return predictions.view(batch_size, self.num_items)
    
    def recommend_for_user(self, user_id: int, num_recommendations: int = 10, 
                          exclude_seen: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a single user
        
        Args:
            user_id: User ID to generate recommendations for
            num_recommendations: Number of recommendations to return
            exclude_seen: Set of item IDs to exclude from recommendations
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, score) tuples
        """
        if not hasattr(self, 'num_users') or user_id >= self.num_users or user_id < 0:
            raise ValueError(f"Invalid user_id: {user_id}")
        
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            scores = self.predict_batch_users(user_tensor).squeeze().cpu().numpy()
            
            # Exclude seen items
            if exclude_seen:
                scores[list(exclude_seen)] = float('-inf')
            
            # Get top-N recommendations
            top_indices = np.argsort(scores)[::-1][:num_recommendations]
            recommendations = [(int(idx), float(scores[idx])) 
                             for idx in top_indices if scores[idx] != float('-inf')]
        
        return recommendations
    
    def save_model(self, path: str):
        """Save model state dict"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state dict"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def get_parameters_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_device(self, device: Union[str, torch.device]):
        """Set device for the model"""
        self.device = torch.device(device)
        self.to(self.device)
    
    def get_embeddings(self, entity_type: str = 'user') -> torch.Tensor:
        """
        Get embeddings for users or items
        
        Args:
            entity_type: 'user' or 'item'
            
        Returns:
            torch.Tensor: Embedding matrix
        """
        if entity_type == 'user' and hasattr(self, 'user_embedding'):
            return self.user_embedding.weight.data
        elif entity_type == 'item' and hasattr(self, 'item_embedding'):
            return self.item_embedding.weight.data
        else:
            raise ValueError(f"Embeddings for {entity_type} not available")