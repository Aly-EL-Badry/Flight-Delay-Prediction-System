from abc import ABC, abstractmethod
import pandas as pd
class CleaningStrategy(ABC):
    def __init__(self, columns):
        """
        Initializes the CleaningStrategy base class.
        """
        super().__init__()
        self.columns = columns

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the given data according to the strategy's purpose.

        Args:
            data (pd.DataFrame): The DataFrame to handle.

        Returns:
            pd.DataFrame: The handled DataFrame.

        Raises:
            Exception: If an error occurs while handling the data.
        """
        pass