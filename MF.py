"""
Matrix Factorization for Collaborative Filtering Recommendation System
Optimized implementation with clean architecture and essential features.

Based on: Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from .BaseModel import BaseModel
from datagenerator import PointwiseGenerator, PairwiseGenerator

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for MF model parameters"""
    hidden_dim: int = 64
    pointwise: bool = True
    loss_func: str = 'mse'
    learning_rate: float = 0.001
    reg_lambda: float = 0.01
    dropout_rate: float = 0.0
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    embedding_init_std: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= self.reg_lambda <= 1:
            raise ValueError("reg_lambda must be between 0 and 1")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")

class MF(BaseModel):
    """
    Matrix Factorization model for recommendation systems.
    
    Features:
    - Configurable architecture and training
    - Memory-efficient batch processing  
    - Model checkpointing and evaluation
    - Comprehensive prediction methods
    """
    
    def __init__(self, dataset, hparams: Union[Dict, ModelConfig], device, model_name: str = "MF"):
        super().__init__()
        
        # Configuration setup
        self.config = ModelConfig(**hparams) if isinstance(hparams, dict) else hparams
        self.model_name = model_name
        self.device = device
        
        # Dataset validation and setup
        self._validate_dataset(dataset)
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        
        # Build model
        self._build_model()
        self._setup_training()
        
        # Training metrics
        self.training_metrics = {
            'epoch_losses': [],
            'validation_scores': [],
            'training_times': []
        }
        
        self.to(device)
        logger.info(f"Initialized {model_name} with {self._count_parameters():,} parameters")

    def _validate_dataset(self, dataset):
        """Validate dataset has required attributes"""
        required_attrs = ['num_users', 'num_items']
        for attr in required_attrs:
            if not hasattr(dataset, attr) or getattr(dataset, attr) <= 0:
                raise ValueError(f"Dataset must have positive {attr}")

    def _build_model(self):
        """Build embedding layers and initialize weights"""
        # Embedding layers with sparse gradients for memory efficiency
        self.user_embedding = nn.Embedding(self.num_users, self.config.hidden_dim, sparse=True)
        self.item_embedding = nn.Embedding(self.num_items, self.config.hidden_dim, sparse=True)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Initialize embeddings
        for embedding in [self.user_embedding, self.item_embedding]:
            nn.init.normal_(embedding.weight, 0.0, self.config.embedding_init_std)
        
        # Loss function
        loss_functions = {
            'mse': F.mse_loss,
            'bce': F.binary_cross_entropy_with_logits,
            'mae': F.l1_loss
        }
        self.loss_func = loss_functions[self.config.loss_func]

    def _setup_training(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute user-item interactions"""
        # Get embeddings with dropout
        user_emb = self.dropout(self.user_embedding(user_ids))
        item_emb = self.dropout(self.item_embedding(item_ids))
        
        # Compute dot product
        return torch.sum(user_emb * item_emb, dim=1)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None) -> Dict:
        """Train the model with monitoring and evaluation"""
        train_matrix = dataset.train_data
        batch_generator = self._create_batch_generator(train_matrix, exp_config)
        
        best_score = None
        
        for epoch in range(1, exp_config.num_epochs + 1):
            # Training step
            epoch_loss = self._train_epoch(batch_generator, exp_config)
            
            # Evaluation step
            epoch_metrics = {'loss': epoch_loss}
            if self._should_evaluate(epoch, exp_config, evaluator):
                eval_scores = evaluator.evaluate(self)
                epoch_metrics.update(eval_scores)
                self.training_metrics['validation_scores'].append(eval_scores)
                
                # Early stopping check
                if early_stop:
                    is_update, should_stop = early_stop.step(eval_scores, epoch)
                    if should_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    if is_update:
                        best_score = eval_scores
            
            # Learning rate scheduling
            self.scheduler.step(epoch_loss)
            
            # Logging
            if loggers:
                for logger_obj in loggers:
                    logger_obj.log_metrics(epoch_metrics, epoch=epoch)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
        
        # Save final checkpoint
        if hasattr(exp_config, 'save_path') and exp_config.save_path:
            self.save_checkpoint(exp_config.save_path, epoch, best_score)
        
        return {'scores': best_score or epoch_metrics, 'final_epoch': epoch}

    def _create_batch_generator(self, train_matrix, exp_config):
        """Create appropriate batch generator"""
        generator_class = PointwiseGenerator if self.config.pointwise else PairwiseGenerator
        
        if self.config.pointwise:
            return generator_class(
                train_matrix, return_rating=True, num_negatives=1,
                batch_size=exp_config.batch_size, shuffle=True, device=self.device
            )
        else:
            return generator_class(
                train_matrix, num_negatives=1, num_positives_per_user=1,
                batch_size=exp_config.batch_size, shuffle=True, device=self.device
            )

    def _train_epoch(self, batch_generator, exp_config) -> float:
        """Train one epoch and return average loss"""
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_users, batch_items, batch_ratings in batch_generator:
            self.optimizer.zero_grad()
            
            # Compute loss
            batch_loss = self._compute_batch_loss(batch_users, batch_items, batch_ratings)
            
            # Add regularization
            reg_loss = self.config.reg_lambda * sum(torch.norm(p) for p in self.parameters())
            total_loss_tensor = batch_loss + reg_loss
            
            # Backward pass with gradient clipping
            total_loss_tensor.backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            total_loss += total_loss_tensor.item()
            num_batches += 1
            
            # Verbose logging
            if exp_config.verbose and num_batches % 100 == 0:
                logger.debug(f'Batch {num_batches}: Loss = {total_loss_tensor.item():.4f}')
        
        avg_loss = total_loss / max(num_batches, 1)
        self.training_metrics['epoch_losses'].append(avg_loss)
        return avg_loss

    def _compute_batch_loss(self, users: torch.Tensor, items: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch"""
        if self.config.pointwise:
            predictions = self.forward(users, items)
            return self.loss_func(predictions, ratings.float())
        else:
            # Pairwise BPR loss
            pos_predictions = self.forward(users, items)
            neg_predictions = self.forward(users, ratings)  # ratings contains negative items for pairwise
            return -F.logsigmoid(pos_predictions - neg_predictions).mean()

    def _should_evaluate(self, epoch: int, exp_config, evaluator) -> bool:
        """Check if evaluation should be performed"""
        return (evaluator is not None and 
                epoch >= getattr(exp_config, 'test_from', 1) and 
                epoch % getattr(exp_config, 'test_step', 1) == 0)

    def predict_batch_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Predict ratings for a batch of users across all items"""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding.weight
        
        # Memory-efficient computation for large catalogs
        if self.num_items > 50000:
            return self._predict_chunked(user_emb, item_emb)
        else:
            return user_emb @ item_emb.T

    def _predict_chunked(self, user_emb: torch.Tensor, item_emb: torch.Tensor, chunk_size: int = 5000) -> torch.Tensor:
        """Memory-efficient prediction using chunking"""
        predictions = []
        for i in range(0, self.num_items, chunk_size):
            end_idx = min(i + chunk_size, self.num_items)
            chunk_pred = user_emb @ item_emb[i:end_idx].T
            predictions.append(chunk_pred)
        return torch.cat(predictions, dim=1)

    def predict(self, eval_users: np.ndarray, eval_pos: np.ndarray, test_batch_size: int) -> np.ndarray:
        """Generate predictions for evaluation"""
        self.eval()
        pred_matrix = np.zeros(eval_pos.shape, dtype=np.float32)
        
        with torch.no_grad():
            for i in range(0, len(eval_users), test_batch_size):
                batch_users = eval_users[i:i + test_batch_size]
                user_tensor = torch.LongTensor(batch_users).to(self.device)
                batch_pred = self.predict_batch_users(user_tensor)
                pred_matrix[batch_users] = batch_pred.cpu().numpy()
        
        # Mask positive interactions
        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix

    def recommend_for_user(self, user_id: int, num_recommendations: int = 10, 
                          exclude_seen: Optional[set] = None) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a user"""
        if not 0 <= user_id < self.num_users:
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

    def save_checkpoint(self, path: str, epoch: int, scores: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'scores': scores,
            'config': asdict(self.config),
            'training_metrics': self.training_metrics,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', {})
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def export_embeddings(self, output_dir: str = "embeddings"):
        """Export user and item embeddings"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export embeddings
        np.save(output_path / "user_embeddings.npy", self.user_embedding.weight.detach().cpu().numpy())
        np.save(output_path / "item_embeddings.npy", self.item_embedding.weight.detach().cpu().numpy())
        
        # Export metadata
        metadata = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'hidden_dim': self.config.hidden_dim,
            'model_name': self.model_name
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Embeddings exported to {output_path}")

    def get_model_info(self) -> Dict:
        """Get model information summary"""
        return {
            'model_name': self.model_name,
            'num_parameters': self._count_parameters(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'config': asdict(self.config),
            'device': str(self.device)
        }