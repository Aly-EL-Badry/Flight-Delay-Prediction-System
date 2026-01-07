import logging
from typing import Optional, List
import pandas as pd
from zenml import step
import mlflow

def readCsv(path: str, **kwargs) -> pd.DataFrame:
    """Simple CSV reader wrapper (keeps step thin)."""
    return pd.read_csv(path, **kwargs)

def coerceBytesAndObjectsToStr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    keepNa: bool = True,
) -> pd.DataFrame:
    """
    Convert bytes and non-string object values in specified columns (or all object columns)
    into strings, preserving NaN by default.
    """
    df = df.copy()
    if columns is None:
        columns = [c for c, dt in df.dtypes.items() if dt == "object"]

    for col in columns:
        if col not in df.columns:
            continue

        def _safeCast(val):
            if pd.isna(val):
                return val if keepNa else ""
            if isinstance(val, (bytes, bytearray)):
                try:
                    return val.decode("utf-8")
                except Exception:
                    return val.decode(errors="replace")
            if isinstance(val, str):
                return val
            return str(val)

        df[col] = df[col].apply(_safeCast)

    return df

def validateMinimal(df: pd.DataFrame, requiredColumns: Optional[List[str]] = None):
    """Quick minimal validation to fail early if critical columns are missing."""
    if requiredColumns:
        missing = [c for c in requiredColumns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    if "ORIGIN_AIRPORT" in df.columns and df["ORIGIN_AIRPORT"].isna().all():
        raise ValueError("Column ORIGIN_AIRPORT is present but contains only null values.")

@step(enable_cache=False)
def dataIngestionStep(DATA_PATH: str) -> pd.DataFrame:
    """
    Ingest raw CSV and return cleaned DataFrame with predictable string-like columns.
    """
    try:
        logging.info("Starting data ingestion from: %s", DATA_PATH)

        df = readCsv(DATA_PATH)

        try:
            mlflow.log_artifact(DATA_PATH)
        except Exception as e:
            logging.warning("mlflow.log_artifact failed, continuing. Error: %s", e)

        df = coerceBytesAndObjectsToStr(df, columns=None, keepNa=True)

        validateMinimal(df, requiredColumns=["ORIGIN_AIRPORT"])

        logging.info("Data ingestion completed. Shape: %s", df.shape)
        return df

    except FileNotFoundError:
        logging.error("CSV file not found at: %s", DATA_PATH)
        raise
    except Exception as e:
        logging.error("Unexpected error in data ingestion: %s", e)
        raise
