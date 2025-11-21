import logging
import pandas as pd
from zenml import step
import mlflow

# ------------------- Flights Step -------------------
@step(enable_cache=False)
def ingest_flights_step(flights_path: str) -> pd.DataFrame:
    """Ingest flights data."""
    try:
        logging.info(f" Reading flights data from: {flights_path}")
        # Read with low_memory=False to avoid mixed-type inference and
        # coerce airport identifier columns to strings to prevent Parquet
        # conversion errors later (pyarrow expects bytes for object cols).
        flights = pd.read_csv(
            flights_path,
            low_memory=False,
            dtype={
                # ensure airport codes are strings even if some rows are numeric
                "ORIGIN_AIRPORT": str,
                "DESTINATION_AIRPORT": str,
            },
        )

        # As a safeguard, explicitly convert these columns to string if present
        for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
            if col in flights.columns:
                flights[col] = flights[col].astype(str)

        mlflow.log_artifact(flights_path)
        logging.info(f" Flights data shape: {flights.shape}")
        return flights
    except FileNotFoundError:
        logging.error(f" Flights file not found: {flights_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while ingesting flights data: {e}")
        raise


# ------------------- Airlines Step -------------------
@step(enable_cache=False)
def ingest_airlines_step(airlines_path: str) -> pd.DataFrame:
    """Ingest airlines data."""
    try:
        logging.info(f" Reading airlines data from: {airlines_path}")
        airlines = pd.read_csv(airlines_path)
        mlflow.log_artifact(airlines_path)
        logging.info(f" Airlines data shape: {airlines.shape}")
        return airlines
    except FileNotFoundError:
        logging.error(f" Airlines file not found: {airlines_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while ingesting airlines data: {e}")
        raise


# ------------------- Airports Step -------------------
@step(enable_cache=False)
def ingest_airports_step(airports_path: str) -> pd.DataFrame:
    """Ingest airports data."""
    try:
        logging.info(f" Reading airports data from: {airports_path}")
        airports = pd.read_csv(airports_path)
        mlflow.log_artifact(airports_path)
        logging.info(f" Airports data shape: {airports.shape}")
        return airports
    except FileNotFoundError:
        logging.error(f" Airports file not found: {airports_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while ingesting airports data: {e}")
        raise
