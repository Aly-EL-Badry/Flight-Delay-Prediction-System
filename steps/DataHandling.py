# cleaning_pipeline_steps_minimal_mlflow.py
from src.CleaningStrategy.base import CleaningStrategy  # abstract parent (not instantiated)
from src.CleaningStrategy.DropStrategy.Columns import DropColumnsStrategy
from src.CleaningStrategy.DropStrategy.InvalidValue import RemoveNumericAirportCodes
from src.CleaningStrategy.DropStrategy.Nulls import DropNullsStrategy

from src.FeatureEngStrategy.ProcessingPipeline import PreprocessingPipeline

from sklearn.model_selection import train_test_split
from typing import List, Tuple
from zenml import step
import pandas as pd
import logging
import os
import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@step()
def DataCleaning(data: pd.DataFrame, dropColumns: List[str], ColsToClean: List[str]) -> pd.DataFrame:
    """
    Minimal MLflow logging:
      - input rows/cols
      - final cleaned rows/cols
    No start/end run here.
    """
    try:
        mlflow.log_param("data_input_rows", int(data.shape[0]))
        mlflow.log_param("data_input_cols", int(data.shape[1]))

        # apply concrete child strategies sequentially
        strategies: List[CleaningStrategy] = [
            DropColumnsStrategy(dropColumns),
            RemoveNumericAirportCodes(),
            DropNullsStrategy(ColsToClean)
        ]

        cleaned = data
        for strat in strategies:
            cleaned = strat.handle_data(cleaned)


        mlflow.log_metric("data_cleaned_rows", int(cleaned.shape[0]))
        mlflow.log_metric("data_cleaned_cols", int(cleaned.shape[1]))

        return cleaned

    except Exception as e:
        logger.exception("Error in DataCleaning: %s", e)
        if mlflow.active_run() is not None:
            mlflow.set_tag("error.DataCleaning", str(e))
        raise


@step
def fitPreprocessingPipeline(
    data: pd.DataFrame,
    LabelCols: list,
    OheCols: list,
    ScaleCols: list,
    CycleCols: list,
    PATH: str
) -> PreprocessingPipeline:
    """
    Fit and save preprocessing pipeline.
    Minimal MLflow logging:
      - counts of column groups
      - logs pipeline artifact if saved
    """
    try:

        mlflow.log_param("n_LabelCols", len(LabelCols) if LabelCols else 0)
        mlflow.log_param("n_OheCols", len(OheCols) if OheCols else 0)
        mlflow.log_param("n_ScaleCols", len(ScaleCols) if ScaleCols else 0)
        mlflow.log_param("n_CycleCols", len(CycleCols) if CycleCols else 0)

        pipeline = PreprocessingPipeline(LabelCols, OheCols, ScaleCols, CycleCols)
        logger.info("start fitting pipeline")
        pipeline.fit(data)
        logger.info("end fitting pipeline")

        logger.info("start saving the pipeline")
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        pipeline.Save(PATH)
        logger.info("end saving the pipeline")


        try:
            mlflow.log_artifact(PATH, artifact_path="preprocessing_pipeline")
        except Exception as e:
            logger.warning("Failed to log pipeline artifact: %s", e)

        return pipeline

    except Exception as e:
        logger.exception("Error in fitPreprocessingPipeline: %s", e)
        if mlflow.active_run() is not None:
            mlflow.set_tag("error.fitPreprocessingPipeline", str(e))
        raise


@step
def applyPreprocessingPipeline(
    pipeline: PreprocessingPipeline,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply pipeline and log only processed row count (minimal).
    """
    try:
        logger.info("start transforming data")
        processed = pipeline.transform(data)
        logger.info("ended transforming data")

        mlflow.log_metric("processed_rows", int(processed.shape[0]))

        return processed

    except Exception as e:
        logger.exception("Error in applyPreprocessingPipeline: %s", e)
        mlflow.set_tag("error.applyPreprocessingPipeline", str(e))
        raise


@step
def SpiltStep(
    data: pd.DataFrame,
    target: str = "weather",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Minimal logging:
      - train/test sizes (rows)
      - saves test CSV and logs it as one artifact
    """
    try:
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        test_data_path = os.path.join("data", "test", "test_data.csv")
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)

        mlflow.log_metric("train_rows", int(X_train.shape[0]))
        mlflow.log_metric("test_rows", int(X_test.shape[0]))
        try:
            mlflow.log_artifact(test_data_path, artifact_path="splits")
        except Exception as e:
            logger.warning("Failed to log test CSV: %s", e)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception("Error in SpiltStep: %s", e)
        mlflow.set_tag("error.SpiltStep", str(e))
        raise
