import numpy as np
import pandas as pd
import joblib
from ..base import BaseFeatureEngStrategy

class CyclicEncodingStrategy(BaseFeatureEngStrategy):
    """
    Converts temporal columns like MONTH, DAY, DAY_OF_WEEK
    into cyclic sin/cos features.

    Example:
        MONTH  → MONTH_sin, MONTH_cos
    """

    def __init__(self, cols):
        """
        cols must contain the names of cyclic columns,
        e.g. ["MONTH", "DAY", "DAY_OF_WEEK"]
        """
        super().__init__(cols)
        self.periods = {}   

    def fit(self, df: pd.DataFrame):
        """
        Store period (range) for each cyclic column.
        """
        for col in self.columns:
            max_val = df[col].max()
            self.periods[col] = max_val  

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_t = df.copy()

        for col in self.columns:
            period = self.periods[col]

            df_t[f"{col}_sin"] = np.sin(2 * np.pi * df_t[col] / period)
            df_t[f"{col}_cos"] = np.cos(2 * np.pi * df_t[col] / period)

            df_t.drop(columns=[col], inplace=True)  

        return df_t

    def apply(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def save(self, save_path: str):
        """
        Store column list + periods.
        """
        payload = {
            "columns": self.columns,
            "periods": self.periods,
        }
        joblib.dump(payload, save_path)

    def load(self, load_path: str):
        payload = joblib.load(load_path)
        self.columns = payload["columns"]
        self.periods = payload["periods"]
