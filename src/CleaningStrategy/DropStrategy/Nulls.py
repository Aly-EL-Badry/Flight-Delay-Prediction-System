from ..base import CleaningStrategy
import pandas as pd

class DropNullsStrategy(CleaningStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows with null values in the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame to handle.

        Returns:
            pd.DataFrame: The DataFrame with rows containing nulls in the specified columns dropped.

        Raises:
            Exception: If an error occurs while handling the data.
        """
        try:
            cleaned_data = data.dropna(subset=self.columns)
            return cleaned_data
        except Exception as e:
            raise Exception(f"Error while dropping nulls: {e}")
