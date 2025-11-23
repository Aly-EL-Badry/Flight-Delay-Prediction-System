from ..base import CleaningStrategy
import pandas as pd

class DropColumnsStrategy(CleaningStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to handle.

        Returns:
            pd.DataFrame: The DataFrame with rows containing nulls in the specified columns dropped.

        Raises:
            Exception: If an error occurs while handling the data.
        """
        try:
            data = data.drop(columns=self.columns, axis=1, errors='ignore')
            return data
        except Exception as e:
            raise Exception(f"Error dropping columns {self.columns}: {e}")