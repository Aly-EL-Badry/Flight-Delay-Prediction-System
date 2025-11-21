import logging
import pandas as pd
import numpy as np
from zenml import step
from src.model_training.evaluate_model import evaluate_model
from typing import Dict, Tuple
import mlflow


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def evaluate_model_step(
    model: object,
    df: pd.DataFrame,
    scaler: object,
    metrics_save_path: str = '../../artifacts/metrics/model_metrics.json'
) -> Dict[str, float]:
    """
    ZenML step for evaluating the trained model.
    
    Args:
        model: Trained Keras model
        df: Preprocessed dataframe
        scaler: Fitted StandardScaler
        metrics_save_path: Path to save evaluation metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        logging.info("Starting model evaluation step...")

        # ZenML's MLflow experiment tracker handles run contexts for steps,
        # so avoid manually starting runs here. Prepare data, scale and evaluate.
        # Prepare test data
        X = df.drop('ARRIVAL_DELAY', axis=1)
        y = df['ARRIVAL_DELAY']

        # Scale features
        columns_to_scale = [
            "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", 
            "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_DEPARTURE",
            "DEPARTURE_TIME", "SCHEDULED_ARRIVAL"
        ]
        columns_to_scale = [col for col in columns_to_scale if col in X.columns]

        X[columns_to_scale] = scaler.transform(X[columns_to_scale])

        # Evaluate model
        metrics = evaluate_model(
            model=model,
            X_test=X.values,
            y_test=y.values,
            metrics_save_path=metrics_save_path
        )

        logging.info("Model evaluation step completed successfully")

        return metrics
            
    except Exception as e:
        logging.error(f"Error in model evaluation step: {e}")
        raise