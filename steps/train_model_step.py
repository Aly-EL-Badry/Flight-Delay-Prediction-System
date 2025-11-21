import logging
import pandas as pd
from zenml import step
from src.model_training.train_model import train_flight_delay_model
from typing import Tuple, Dict
import mlflow
from sklearn.preprocessing import StandardScaler


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model_step(
    df: pd.DataFrame,
    model_save_path: str = '../../artifacts/models/flight_delay_model.keras'
) -> Tuple[object, Dict[str, float], StandardScaler]:
    """
    ZenML step for training the flight delay prediction model.
    
    Args:
        df: Preprocessed and feature-engineered dataframe
        model_save_path: Path to save the trained model
    
    Returns:
        Trained model, evaluation metrics, and scaler
    """
    try:
        logging.info("Starting model training step...")

        # The ZenML MLflow experiment tracker manages MLflow run contexts
        # for each step. Avoid manually starting a run here to prevent
        # conflicts (e.g., 'run already active'). Just log params/metrics
        # directly and train the model.
        mlflow.log_param("dataset_shape", df.shape)
        mlflow.log_param("dataset_columns", list(df.columns))

        # Train model
        model, metrics, scaler = train_flight_delay_model(
            df=df,
            model_save_path=model_save_path
        )

        logging.info("Model training step completed successfully")

        return model, metrics, scaler
            
    except Exception as e:
        logging.error(f"Error in model training step: {e}")
        raise