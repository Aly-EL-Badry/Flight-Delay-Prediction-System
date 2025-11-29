import logging
from zenml import pipeline
from steps.DataIngestion import (
    ingest_flights_step,
    ingest_airports_step,
    ingest_airlines_step
)
from steps.DataHandling import clean__df
from steps.train_model_step import train_model_step
from steps.evaluation import evaluate_model_step


@pipeline(enable_cache=False)
def flight_delay_training_pipeline(
    flights_path: str,
    airports_path: str,
    airlines_path: str,
    model_save_path: str = '../../artifacts/models/flight_delay_model.keras',
    metrics_save_path: str = '../../artifacts/metrics/model_metrics.json'
):
    """
    Complete training pipeline for flight delay prediction model.
    
    Steps:
        1. Ingest raw data (flights, airports, airlines)
        2. Clean and preprocess data
        3. Feature engineering
        4. Train model
        5. Evaluate model
    
    Args:
        flights_path: Path to flights CSV file
        airports_path: Path to airports CSV file
        airlines_path: Path to airlines CSV file
        model_save_path: Path to save trained model
        metrics_save_path: Path to save evaluation metrics
    """
    logging.info("=" * 60)
    logging.info("Starting Flight Delay Training Pipeline")
    logging.info("=" * 60)
    
    # Step 1: Ingest raw data
    logging.info("Step 1: Data Ingestion")
    flights_df = ingest_flights_step(flights_path)
    airports_df = ingest_airports_step(airports_path)
    airlines_df = ingest_airlines_step(airlines_path)
    
    # Step 2 & 3: Clean data and feature engineering
    logging.info("Step 2 & 3: Data Cleaning and Feature Engineering")
    processed_df = clean__df(flights_df, airports_df)
    
    # Step 4: Train model
    logging.info("Step 4: Model Training")
    model, train_metrics, scaler = train_model_step(
        df=processed_df,
        model_save_path=model_save_path
    )
    
    # Step 5: Evaluate model
    logging.info("Step 5: Model Evaluation")
    eval_metrics = evaluate_model_step(
        model=model,
        df=processed_df,
        scaler=scaler,
        metrics_save_path=metrics_save_path
    )
    
    logging.info("=" * 60)
    logging.info("Flight Delay Training Pipeline Completed Successfully")
    logging.info("=" * 60)
    
    return model, eval_metrics


if __name__ == "__main__":
    # Run the pipeline
    pipeline_run = flight_delay_training_pipeline(
        flights_path="../../data/flights.csv",
        airports_path="../../data/airports.csv",
        airlines_path="../../data/airlines.csv"
    )