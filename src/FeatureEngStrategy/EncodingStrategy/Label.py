from ..base import BaseFeatureEngStrategy
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os
import numpy as np

class LabelEncodingStrategy(BaseFeatureEngStrategy):
    def __init__(self, cols):
        super().__init__(cols)
        self.encoders = {col: LabelEncoder() for col in cols}

    def fit(self, df: pd.DataFrame):
        for col in self.columns:
            self.encoders[col].fit(df[col].astype(str))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()

        for col in self.columns:
            encoder = self.encoders[col]
            values = df_encoded[col].astype(str).values
            encoded = np.full(values.shape[0], -1, dtype=int)
            known_mask = np.isin(values, encoder.classes_)
            encoded[known_mask] = encoder.transform(values[known_mask])
            df_encoded.loc[:, col] = encoded

        return df_encoded


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {"encoders": self.encoders, "columns": list(self.columns)}
        joblib.dump(payload, save_path)

    def load(self, load_path: str):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Encoders bundle not found: {load_path}")
        payload = joblib.load(load_path)
        self.encoders = payload["encoders"]
        self.columns = tuple(payload["columns"])
