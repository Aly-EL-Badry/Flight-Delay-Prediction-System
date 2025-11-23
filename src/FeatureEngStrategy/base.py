from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureEngStrategy(ABC):
    """
    Abstract base class for all feature engineering strategies.

    This class defines a unified interface that every feature 
    engineering component must follow to ensure:
      - Clean software design (Strategy Pattern)
      - Compatibility with ML pipelines (training + inference)
      - Easy integration with tools like ZenML

    Each subclass MUST implement:
        • fit(df):      Learn metadata/statistics from training data
        • transform(df): Apply the learned transformation
        • apply(df):     Used ONLY during training (fit + transform)
        • save(path):    Persist any learned metadata for inference
        • load(path):    Load saved metadata into the transformer

    Attributes
    ----------
    columns : list
        The target columns that the strategy operates on.
    """

    def __init__(self, cols):
        """
        Initialize the strategy with a list of columns.

        Parameters
        ----------
        cols : list
            Columns to be processed by the strategy.
        """
        self.columns = cols

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Learn metadata/statistics from training data.
        To be executed only during training.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        """
        Apply transformation using previously learned metadata.
        Used in both training and inference.
        """
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full training-time execution = fit + transform.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        pass

    @abstractmethod
    def save(self, save_path: str):
        """
        Save all learned metadata (encoders, stats, scalers...).

        Parameters
        ----------
        save_path : str
            File path to save metadata.
        """
        pass

    @abstractmethod
    def load(self, load_path: str):
        """
        Load metadata previously saved during training.

        Parameters
        ----------
        load_path : str
            File path where metadata was saved.
        """
        pass
