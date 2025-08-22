"""
Recommendation System - Main Training Script
Matrix Factorization for Amazon Review Datasets

This script provides complete functionality for:
1. Loading and preprocessing Amazon review datasets
2. Training Matrix Factorization models with various configurations
3. Model evaluation and recommendation generation
4. Hyperparameter tuning and experimentation
5. Results visualization and analysis

Author: Athena Project Team
Date: June 2025
"""

import os
import sys
import argparse
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Local imports
from MF_ import MF, ModelConfig
from BaseModel import BaseModel
from Amazon_data_process import AmazonDataset
from datagenerator import DataLoader, PointwiseGenerator, PairwiseGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration class for training experiments"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def step(self, score: float, epoch: int) -> Tuple[bool, bool]:
        """
        Args:
            score: Current validation score (higher is better)
            epoch: Current epoch number
            
        Returns:
            Tuple of (is_best, should_stop)
        """
        if self.best_score is None:
            self.best_score = score
            return True, False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            return False, self.counter >= self.patience

class Evaluator:
    """Model evaluator with various metrics"""
    def __init__(self, dataset: AmazonDataset, k_list: List[int] = [5, 10, 20]):
        self.dataset = dataset
        self.k_list = k_list
        self.test_users = np.arange(dataset.num_users)
        self.test_items = dataset.test_data
        
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """
        Evaluate model performance using various metrics
        
        Args:
            model: Trained recommendation model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Generate predictions
        predictions = model.predict(
            eval_users=self.test_users,
            eval_pos=self.dataset.train_data.toarray(),
            test_batch_size=1024
        )
        
        metrics = {}
        
        # Calculate ranking metrics for each k
        for k in self.k_list:
            ndcg_k = self._calculate_ndcg_at_k(predictions, k)
            recall_k = self._calculate_recall_at_k(predictions, k)
            precision_k = self._calculate_precision_at_k(predictions, k)
            
            metrics[f'NDCG@{k}'] = ndcg_k
            metrics[f'Recall@{k}'] = recall_k
            metrics[f'Precision@{k}'] = precision_k
        
        # Calculate additional metrics
        metrics['Coverage'] = self._calculate_coverage(predictions)
        metrics['Popularity_Bias'] = self._calculate_popularity_bias(predictions)
        
        logger.info("Evaluation completed")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_ndcg_at_k(self, predictions: np.ndarray, k: int) -> float:
        """Calculate NDCG@k metric"""
        ndcg_scores = []
        
        for user_idx in range(self.dataset.num_users):
            # Get test items for this user
            test_items = self.test_items[user_idx].nonzero()[1]
            if len(test_items) == 0:
                continue
            
            # Get top-k predictions
            user_pred = predictions[user_idx]
            top_k_items = np.argsort(user_pred)[::-1][:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            # Calculate IDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_recall_at_k(self, predictions: np.ndarray, k: int) -> float:
        """Calculate Recall@k metric"""
        recall_scores = []
        
        for user_idx in range(self.dataset.num_users):
            test_items = set(self.test_items[user_idx].nonzero()[1])
            if len(test_items) == 0:
                continue
            
            user_pred = predictions[user_idx]
            top_k_items = set(np.argsort(user_pred)[::-1][:k])
            
            recall = len(test_items & top_k_items) / len(test_items)
            recall_scores.append(recall)
        
        return np.mean(recall_scores) if recall_scores else 0.0
    
    def _calculate_precision_at_k(self, predictions: np.ndarray, k: int) -> float:
        """Calculate Precision@k metric"""
        precision_scores = []
        
        for user_idx in range(self.dataset.num_users):
            test_items = set(self.test_items[user_idx].nonzero()[1])
            if len(test_items) == 0:
                continue
            
            user_pred = predictions[user_idx]
            top_k_items = set(np.argsort(user_pred)[::-1][:k])
            
            precision = len(test_items & top_k_items) / k if k > 0 else 0.0
            precision_scores.append(precision)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    def _calculate_coverage(self, predictions: np.ndarray) -> float:
        """Calculate catalog coverage"""
        recommended_items = set()
        
        for user_idx in range(min(1000, self.dataset.num_users)):  # Sample for efficiency
            user_pred = predictions[user_idx]
            top_items = np.argsort(user_pred)[::-1][:20]
            recommended_items.update(top_items)
        
        return len(recommended_items) / self.dataset.num_items
    
    def _calculate_popularity_bias(self, predictions: np.ndarray) -> float:
        """Calculate popularity bias (lower is better)"""
        # Calculate item popularity from training data
        item_popularity = np.array(self.dataset.train_data.sum(axis=0)).flatten()
        
        total_bias = 0.0
        num_users = 0
        
        for user_idx in range(min(1000, self.dataset.num_users)):  # Sample for efficiency
            user_pred = predictions[user_idx]
            top_items = np.argsort(user_pred)[::-1][:10]
            
            # Calculate average popularity of recommended items
            avg_popularity = np.mean(item_popularity[top_items])
            total_bias += avg_popularity
            num_users += 1
        
        return total_bias / num_users if num_users > 0 else 0.0

class RecommendationTrainer:
    """
    Main trainer class for recommendation systems.
    Handles dataset loading, model training, evaluation, and analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize recommendation trainer with configuration.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'embeddings').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        logger.info(f"Recommendation trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_dataset(self) -> AmazonDataset:
        """Load and preprocess Amazon dataset"""
        dataset_config = self.config.get('dataset', {})
        
        logger.info("Loading Amazon dataset...")
        dataset = AmazonDataset(
            data_path=dataset_config.get('data_path', 'data/amazon_books.json.gz'),
            dataset_name=dataset_config.get('name', 'books'),
            min_interactions=dataset_config.get('min_interactions', 5),
            test_ratio=dataset_config.get('test_ratio', 0.2)
        )
        
        # Print and save dataset statistics
        stats = dataset.get_statistics()
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value:,}")
        
        # Save dataset statistics
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save processed dataset for future use
        if dataset_config.get('save_processed', True):
            dataset.save_processed_data(str(self.output_dir / 'processed_data'))
            logger.info("Processed dataset saved")
        
        return dataset
    
    def create_model(self, dataset: AmazonDataset) -> MF:
        """Create and initialize Matrix Factorization model"""
        model_config = self.config.get('model', {})
        
        # Create model configuration
        config = ModelConfig(
            hidden_dim=model_config.get('hidden_dim', 128),
            pointwise=model_config.get('pointwise', True),
            loss_func=model_config.get('loss_func', 'mse'),
            learning_rate=model_config.get('learning_rate', 0.001),
            reg_lambda=model_config.get('reg_lambda', 0.01),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            weight_decay=model_config.get('weight_decay', 1e-5),
            gradient_clip_norm=model_config.get('gradient_clip_norm', 1.0),
            embedding_init_std=model_config.get('embedding_init_std', 0.1)
        )
        
        # Initialize model
        model_name = f"MF_{dataset.dataset_name}_{config.hidden_dim}d"
        model = MF(dataset, config, self.device, model_name=model_name)
        
        logger.info(f"Model created: {model_name}")
        logger.info(f"Parameters: {model.get_parameters_count():,}")
        logger.info(f"Model configuration: {config.__dict__}")
        
        return model
    
    def train_model(self, model: MF, dataset: AmazonDataset) -> Dict[str, Any]:
        """Train the recommendation model"""
        training_config = self.config.get('training', {})
        
        # Create experiment configuration
        exp_config = ExperimentConfig({
            'num_epochs': training_config.get('num_epochs', 100),
            'batch_size': training_config.get('batch_size', 1024),
            'verbose': training_config.get('verbose', True),
            'test_from': training_config.get('test_from', 5),
            'test_step': training_config.get('test_step', 5)
        })
        
        # Create evaluator
        evaluator = Evaluator(dataset, k_list=training_config.get('k_list', [5, 10, 20]))
        
        # Create early stopping
        early_stop = None
        if training_config.get('early_stopping', True):
            early_stop = EarlyStopping(
                patience=training_config.get('patience', 10),
                min_delta=training_config.get('min_delta', 0.001)
            )
        
        # Training
        logger.info("Starting model training...")
        logger.info(f"Epochs: {exp_config.num_epochs}")
        logger.info(f"Batch size: {exp_config.batch_size}")
        
        start_time = time.time()
        
        try:
            results = model.fit(
                dataset=dataset,
                exp_config=exp_config,
                evaluator=evaluator,
                early_stop=early_stop,
                loggers=None
            )
            
            training_time = time.time() - start_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save model
            model_path = self.output_dir / 'models' / f'{model.model_name}_final.pt'
            model.save_checkpoint(str(model_path), exp_config.num_epochs, results.get('scores'))
            
            # Export embeddings
            embedding_dir = self.output_dir / 'embeddings' / model.model_name
            model.export_embeddings(str(embedding_dir))
            
            # Prepare results
            training_results = {
                'training_time': training_time,
                'final_scores': results.get('scores', {}),
                'final_epoch': results.get('final_epoch', exp_config.num_epochs),
                'model_info': model.get_model_info(),
                'training_config': training_config
            }
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def generate_recommendations(self, model: MF, dataset: AmazonDataset, num_users: int = 10) -> Dict[str, Any]:
        """Generate sample recommendations for analysis"""
        logger.info(f"Generating recommendations for {num_users} users...")
        
        recommendations = {}
        recommendation_stats = {
            'avg_score': [],
            'score_std': [],
            'num_unique_items': set()
        }
        
        for user_idx in range(min(num_users, dataset.num_users)):
            # Get user's training interactions (to exclude)
            user_train_items = set(dataset.get_user_items(user_idx, 'train'))
            
            # Generate recommendations
            user_recommendations = model.recommend_for_user(
                user_id=user_idx,
                num_recommendations=20,
                exclude_seen=user_train_items
            )
            
            # Convert to original item IDs
            original_recommendations = []
            scores = []
            for item_idx, score in user_recommendations:
                original_item_id = dataset.reverse_item_map.get(item_idx, f'item_{item_idx}')
                original_recommendations.append((original_item_id, score))
                scores.append(score)
                recommendation_stats['num_unique_items'].add(item_idx)
            
            recommendations[f'user_{user_idx}'] = {
                'original_user_id': dataset.reverse_user_map.get(user_idx, f'user_{user_idx}'),
                'recommendations': original_recommendations,
                'num_train_interactions': len(user_train_items)
            }
            
            # Statistics
            if scores:
                recommendation_stats['avg_score'].append(np.mean(scores))
                recommendation_stats['score_std'].append(np.std(scores))
            
            # Log top 5 recommendations for first few users
            if user_idx < 3:
                logger.info(f"Top 5 recommendations for user {user_idx}:")
                for i, (item_id, score) in enumerate(original_recommendations[:5]):
                    logger.info(f"  {i+1}. {item_id}: {score:.4f}")
        
        # Calculate recommendation statistics
        rec_stats = {
            'avg_recommendation_score': np.mean(recommendation_stats['avg_score']),
            'avg_score_std': np.mean(recommendation_stats['score_std']),
            'total_unique_items_recommended': len(recommendation_stats['num_unique_items']),
            'catalog_coverage': len(recommendation_stats['num_unique_items']) / dataset.num_items
        }
        
        logger.info("Recommendation Statistics:")
        for key, value in rec_stats.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save recommendations
        output_data = {
            'recommendations': recommendations,
            'statistics': rec_stats,
            'metadata': {
                'model_name': model.model_name,
                'dataset_name': dataset.dataset_name,
                'generation_time': time.time()
            }
        }
        
        with open(self.output_dir / 'recommendations.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        return output_data
    
    def create_visualizations(self, model: MF, dataset: AmazonDataset, training_results: Dict[str, Any]):
        """Create visualizations for analysis"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Training Loss Plot
        if model.training_metrics.get('epoch_losses'):
            plt.figure(figsize=(10, 6))
            plt.plot(model.training_metrics['epoch_losses'])
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(self.output_dir / 'plots' / 'training_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Validation Metrics Plot
        if model.training_metrics.get('validation_scores'):
            metrics_data = {}
            for epoch_scores in model.training_metrics['validation_scores']:
                for metric, value in epoch_scores.items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
            
            if metrics_data:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, (metric, values) in enumerate(metrics_data.items()):
                    if i < 4:  # Show top 4 metrics
                        axes[i].plot(values)
                        axes[i].set_title(f'{metric} Over Time')
                        axes[i].set_xlabel('Evaluation Step')
                        axes[i].set_ylabel(metric)
                        axes[i].grid(True)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'plots' / 'validation_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Dataset Statistics Visualization
        stats = dataset.get_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # User interaction distribution
        user_interactions = np.array(dataset.train_data.sum(axis=1)).flatten()
        axes[0, 0].hist(user_interactions, bins=50, alpha=0.7)
        axes[0, 0].set_title('User Interaction Distribution')
        axes[0, 0].set_xlabel('Number of Interactions')
        axes[0, 0].set_ylabel('Number of Users')
        
        # Item interaction distribution  
        item_interactions = np.array(dataset.train_data.sum(axis=0)).flatten()
        axes[0, 1].hist(item_interactions, bins=50, alpha=0.7)
        axes[0, 1].set_title('Item Interaction Distribution')
        axes[0, 1].set_xlabel('Number of Interactions')
        axes[0, 1].set_ylabel('Number of Items')
        
        # Dataset overview
        overview_data = [stats['num_users'], stats['num_items'], stats['train_interactions']]
        overview_labels = ['Users', 'Items', 'Interactions']
        axes[1, 0].bar(overview_labels, overview_data)
        axes[1, 0].set_title('Dataset Overview')
        axes[1, 0].set_ylabel('Count')
        
        # Sparsity visualization
        sparsity_data = [stats['density'], 1 - stats['density']]
        sparsity_labels = ['Density', 'Sparsity']
        axes[1, 1].pie(sparsity_data, labels=sparsity_labels, autopct='%1.2f%%')
        axes[1, 1].set_title('Data Sparsity')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved to plots directory")
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete recommendation system experiment"""
        logger.info("=" * 60)
        logger.info("STARTING RECOMMENDATION SYSTEM EXPERIMENT")
        logger.info("=" * 60)
        
        experiment_start_time = time.time()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Create model
        model = self.create_model(dataset)
        
        # Train model
        training_results = self.train_model(model, dataset)
        
        # Generate recommendations
        recommendation_results = self.generate_recommendations(model, dataset)
        
        # Create visualizations
        self.create_visualizations(model, dataset, training_results)
        
        # Calculate total experiment time
        total_time = time.time() - experiment_start_time
        
        # Compile final results
        final_results = {
            'experiment_info': {
                'total_time': total_time,
                'dataset_name': dataset.dataset_name,
                'model_name': model.model_name,
                'device': str(self.device),
                'timestamp': time.time()
            },
            'dataset_stats': dataset.get_statistics(),
            'model_info': model.get_model_info(),
            'training_results': training_results,
            'recommendation_results': recommendation_results,
            'config': self.config
        }
        
        # Save final results
        with open(self.output_dir / 'experiment_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Model: {model.model_name}")
        logger.info(f"Parameters: {model.get_parameters_count():,}")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return final_results

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default configuration")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for recommendation system"""
    return {
        "experiment_name": "amazon_recommendation_experiment",
        "output_dir": "outputs",
        
        "dataset": {
            "data_path": "data/amazon_books.json.gz",
            "name": "books",
            "min_interactions": 5,
            "test_ratio": 0.2,
            "save_processed": True
        },
        
        "model": {
            "hidden_dim": 128,
            "pointwise": True,
            "loss_func": "mse",
            "learning_rate": 0.001,
            "reg_lambda": 0.01,
            "dropout_rate": 0.1,
            "weight_decay": 1e-5,
            "gradient_clip_norm": 1.0,
            "embedding_init_std": 0.1
        },
        
        "training": {
            "num_epochs": 100,
            "batch_size": 1024,
            "verbose": True,
            "test_from": 5,
            "test_step": 5,
            "early_stopping": True,
            "patience": 10,
            "min_delta": 0.001,
            "k_list": [5, 10, 20]
        }
    }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Recommendation System - Matrix Factorization Training")
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to Amazon dataset (overrides config)')
    parser.add_argument('--dataset-name', type=str, default=None,
                      help='Dataset name (books, electronics, etc.)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Training batch size (overrides config)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                      help='Model hidden dimension (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                      help='Learning rate (overrides config)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set GPU device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config['dataset']['data_path'] = args.data_path
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    
    # Initialize and run trainer
    trainer = RecommendationTrainer(config)
    
    try:
        results = trainer.run_experiment()
        
        # Print summary
        training_results = results['training_results']
        logger.info("\n" + "=" * 40 + " SUMMARY " + "=" * 40)
        logger.info(f"Dataset: {results['dataset_stats']['num_users']:,} users, {results['dataset_stats']['num_items']:,} items")
        logger.info(f"Training time: {training_results['training_time']:.2f} seconds")
        logger.info(f"Final epoch: {training_results['final_epoch']}")
        
        if 'final_scores' in training_results:
            logger.info("Final scores:")
            for metric, score in training_results['final_scores'].items():
                logger.info(f"  {metric}: {score:.4f}")
        
        logger.info(f"All results saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()