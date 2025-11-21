import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from typing import Tuple, Dict
import mlflow
import mlflow.keras


class FlightDelayModelTrainer:
    """Trainer class for flight delay prediction model."""
    
    def __init__(
        self,
        input_dim: int = 29,
        units: list = [512, 256, 128, 64, 32],
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 20
    ):
        self.input_dim = input_dim
        self.units = units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self) -> Sequential:
        """Build the neural network architecture."""
        logging.info("Building model architecture...")
        
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        
        for u in self.units:
            model.add(Dense(u, activation='elu'))
        
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        logging.info(f"Model built with {len(self.units)} hidden layers")
        return model
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'ARRIVAL_DELAY',
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data for training."""
        logging.info("Preparing data for training...")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Identify columns to scale
        columns_to_scale = [
            "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", 
            "DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_DEPARTURE",
            "DEPARTURE_TIME", "SCHEDULED_ARRIVAL"
        ]
        
        # Only scale columns that exist in the dataframe
        columns_to_scale = [col for col in columns_to_scale if col in X.columns]
        
        # Scale continuous features
        X[columns_to_scale] = self.scaler.fit_transform(X[columns_to_scale])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        logging.info(f"Training set size: {X_train.shape[0]}")
        logging.info(f"Validation set size: {X_val.shape[0]}")
        logging.info(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> keras.callbacks.History:
        """Train the model."""
        logging.info("Starting model training...")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Log parameters to MLflow
        mlflow.log_param("input_dim", self.input_dim)
        mlflow.log_param("units", self.units)
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("epochs", self.epochs)
        
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logging.info("Model training completed")
        
        # Log metrics to MLflow
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_train_mae = self.history.history['mae'][-1]
        final_val_mae = self.history.history['val_mae'][-1]
        
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_train_mae", final_train_mae)
        mlflow.log_metric("final_val_mae", final_val_mae)
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        logging.info("Evaluating model on test data...")
        
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'test_loss': test_results[0],
            'test_mae': test_results[1],
            'test_mse': test_results[2]
        }
        
        logging.info(f"Test MSE: {metrics['test_mse']:.4f}")
        logging.info(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        return metrics
    
    def save_model(self, path: str):
        """Save the trained model."""
        logging.info(f"Saving model to {path}")
        keras.saving.save_model(self.model, path)
        
        # Log model to MLflow
        mlflow.keras.log_model(self.model, "model")
        
        logging.info("Model saved successfully")


def train_flight_delay_model(
    df: pd.DataFrame,
    model_save_path: str = '../../artifacts/models/flight_delay_model.keras'
) -> Tuple[Sequential, Dict[str, float], StandardScaler]:
    """
    Main training function.
    
    Args:
        df: Preprocessed dataframe
        model_save_path: Path to save the trained model
    
    Returns:
        Trained model, evaluation metrics, and scaler
    """
    # Determine input dimension from dataframe
    input_dim = len(df.columns) - 1  # -1 for target column
    
    # Initialize trainer
    trainer = FlightDelayModelTrainer(
        input_dim=input_dim,
        units=[512, 256, 128, 64, 32],
        learning_rate=1e-4,
        batch_size=256,
        epochs=20
    )
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.save_model(model_save_path)
    
    return trainer.model, metrics, trainer.scaler