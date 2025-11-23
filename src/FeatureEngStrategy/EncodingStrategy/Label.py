from ..base import BaseFeatureEngStrategy
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os

class LabelEncodingStrategy(BaseFeatureEngStrategy):
    def __init__(self, cols):
        super().__init__(cols)

        self.encoders = {col: LabelEncoder() for col in cols}

    def _safe_transform(self, col, value):
        """
        Handles unseen labels by assigning -1.
        """
        encoder = self.encoders[col]

        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1  
    
    def fit(self, df: pd.DataFrame):
        """
        Fit LabelEncoders on training data only.
        """
        try:
            for col in self.columns:
                self.encoders[col].fit(df[col].astype(str))

        except Exception as e:
            raise Exception(f"Error fitting Label Encoders: {e}")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using already fitted encoders.
        Works for both training and inference.
        """
        try:
            df_encoded = df.copy()

            for col in self.columns:
                if col not in df.columns:
                    raise Exception(f"Column '{col}' not found in DataFrame")

                df_encoded[col] = df_encoded[col].map(
                    lambda x: self._safe_transform(col, x)
                )

            return df_encoded

        except Exception as e:
            raise Exception(f"Error transforming with Label Encoders: {e}")


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For training stage only.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, save_path: str):
        """
        Save all encoders and metadata as a single artifact.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        payload = {
            "encoders": self.encoders,           
            "columns": list(self.columns),       
        }

        joblib.dump(payload, save_path)

    def load(self, load_path: str):
        """
        Load the single encoders bundle and restore state.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Encoders bundle not found: {load_path}")

        payload = joblib.load(load_path)

        # basic validation
        encoders = payload.get("encoders")
        cols = payload.get("columns")
        if not encoders or not cols:
            raise ValueError("Encoders bundle missing required fields")

        self.encoders = encoders
        self.columns = tuple(cols)  
