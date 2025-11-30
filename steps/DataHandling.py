# Cleaning Lib
from ..src.CleaningStrategy.base import CleaningStrategy
from ..src.CleaningStrategy.DropStrategy.Columns import DropColumnsStrategy
from ..src.CleaningStrategy.DropStrategy.InvalidValue import RemoveNumericAirportCodes
from ..src.CleaningStrategy.DropStrategy.Nulls import DropNullsStrategy

# processing Pipeline
from ..src.FeatureEngStrategy.ProcessingPipeline import PreprocessingPipeline

from sklearn.model_selection import train_test_split
from typing import List, Tuple
from zenml import step
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@step()
def DataCleaning(data : pd.DataFrame, dropColumns : List[str]) -> pd.DataFrame :
    try:
        CleaningStr = CleaningStrategy()
        logger.info("start Cleaning Data")
    

        CleaningStr = DropColumnsStrategy(dropColumns)
        DroppedColumnData = CleaningStr.handle_data(data)

        CleaningStr = RemoveNumericAirportCodes()
        DroppedColumnData = CleaningStr.handle_data(DroppedColumnData)

        CleaningStr = DropNullsStrategy()
        CleanedData = CleaningStr.handle_data(DroppedColumnData)

        logger.info("end Cleaning Data")

        return CleanedData
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

@step
def fitPreprocessingPipeline(
    data: pd.DataFrame,
    LabelCols: list,
    OheCols: list,
    ScaleCols: list,
    PATH: str
) -> PreprocessingPipeline:
    """
    Fit the preprocessing pipeline to the training data.

    Args:
        X_train (pd.DataFrame): Training data
        label_cols (list): Columns to be label encoded
        ohe_cols (list): Columns to be one-hot encoded
        scale_cols (list): Columns to be scaled

    Returns:
        PreprocessingPipeline: The fitted preprocessing pipeline
    """
    try: 
        logger.info("start training the pipeline")

        pipeline = PreprocessingPipeline(LabelCols, OheCols, ScaleCols)
        pipeline.fit(data)
        pipeline.Save(PATH)

        logger.info("End training the pipeline")
        return pipeline
    except Exception as e:
        logger.error(f"Error in Processing: {e}")

@step
def applyPreprocessingPipeline(
    pipeline: PreprocessingPipeline,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies the preprocessing pipeline to the given data.

    Args:
        pipeline (PreprocessingPipeline): The preprocessing pipeline to apply.
        data (pd.DataFrame): The data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    dataProcessed = pipeline.transform(data)
    return dataProcessed


@step
def SpiltStep(
    data: pd.DataFrame,
    target: str = "weather",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a given dataset into training and test sets using train_test_split.
    
    Args:
        data (pd.DataFrame): The dataset to split.
        target (str, optional): The target column name. Defaults to "weather".
        test_size (float, optional): The proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
            Defaults to 42.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training data, test data, 
        training labels, and test labels.
    """
    try:
        logger.info(f"Splitting data with test_size={test_size}")
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # save the data of the test in a test file 
        test_data_path = os.path.join("data", "test", "test_data.csv")
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in data splitting: {e}")
        raise


        