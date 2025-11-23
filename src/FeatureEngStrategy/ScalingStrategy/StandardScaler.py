import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from ..base import BaseFeatureEngStrategy

class ScalingStrategy(BaseFeatureEngStrategy):
    """
    Scales continuous numeric features using StandardScaler.
    """

    def __init__(self, cols):
        super().__init__(cols)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        """
        Learn scaling parameters.
        """
        self.scaler.fit(df[self.columns])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_t = df.copy()
        df_t[self.columns] = self.scaler.transform(df_t[self.columns])
        return df_t

    def apply(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def save(self, save_path: str):
        payload = {
            "columns": self.columns,
            "scaler": self.scaler
        }
        joblib.dump(payload, save_path)

    def load(self, load_path: str):
        payload = joblib.load(load_path)
        self.columns = payload["columns"]
        self.scaler = payload["scaler"]
