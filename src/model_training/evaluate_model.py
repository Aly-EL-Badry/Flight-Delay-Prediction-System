import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import mlflow
import json
import os


class ModelEvaluator:
    """Class for evaluating flight delay prediction model."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        logging.info("Calculating evaluation metrics...")
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        self.metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        logging.info(f"MSE: {mse:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"R² Score: {r2:.4f}")
        logging.info(f"MAPE: {mape:.2f}%")
        
        return self.metrics
    
    def calculate_accuracy_within_threshold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 15.0
    ) -> float:
        """
        Calculate percentage of predictions within a certain threshold.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            threshold: Acceptable error threshold in minutes
        
        Returns:
            Percentage of predictions within threshold
        """
        errors = np.abs(y_true - y_pred)
        within_threshold = np.sum(errors <= threshold) / len(errors) * 100
        
        logging.info(f"Predictions within {threshold} minutes: {within_threshold:.2f}%")
        self.metrics[f'accuracy_within_{int(threshold)}min'] = float(within_threshold)
        
        return within_threshold
    
    def analyze_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze error distribution.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Dictionary of error statistics
        """
        errors = y_true - y_pred
        
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'median_error': float(np.median(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75))
        }
        
        logging.info(f"Mean error: {error_stats['mean_error']:.4f}")
        logging.info(f"Std error: {error_stats['std_error']:.4f}")
        
        self.metrics.update(error_stats)
        
        return error_stats
    
    def save_metrics(self, path: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logging.info(f"Metrics saved to {path}")
    
    def log_to_mlflow(self):
        """Log all metrics to MLflow."""
        for key, value in self.metrics.items():
            mlflow.log_metric(key, value)
        
        logging.info("Metrics logged to MLflow")


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics_save_path: str = '../../artifacts/metrics/model_metrics.json'
) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test targets
        metrics_save_path: Path to save metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    logging.info("Starting model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    
    # Calculate accuracy within thresholds
    for threshold in [10, 15, 30]:
        evaluator.calculate_accuracy_within_threshold(y_test, y_pred, threshold)
    
    # Analyze error distribution
    evaluator.analyze_error_distribution(y_test, y_pred)
    
    # Save metrics
    evaluator.save_metrics(metrics_save_path)
    
    # Log to MLflow
    evaluator.log_to_mlflow()
    
    logging.info("Model evaluation completed")
    
    return evaluator.metrics


def compare_models(
    model1_metrics: Dict[str, float],
    model2_metrics: Dict[str, float],
    primary_metric: str = 'mae'
) -> str:
    """
    Compare two models based on metrics.
    
    Args:
        model1_metrics: Metrics from first model
        model2_metrics: Metrics from second model
        primary_metric: Primary metric for comparison
    
    Returns:
        Name of better model
    """
    if model1_metrics[primary_metric] < model2_metrics[primary_metric]:
        logging.info("Model 1 performs better")
        return "model1"
    else:
        logging.info("Model 2 performs better")
        return "model2"