from ..base import BaseFeatureEngStrategy
import pandas as pd
import joblib
import os

class OneHotEncodingStrategy(BaseFeatureEngStrategy):
    def __init__(self, cols):
        super().__init__(cols)
        self.categories_ = {col: [] for col in cols}

    def fit(self, df: pd.DataFrame):
        """
        Fit the one-hot encoder on training data.
        Stores the categories for each column.
        """
        try:
            for col in self.columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                    
                self.categories_[col] = sorted(df[col].dropna().astype(str).unique())
        except Exception as e:
            raise Exception(f"Error fitting One-Hot Encoder: {e}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using the fitted categories.
        Unseen categories will be ignored (all zeros).
        """
        try:
            df_encoded = df.copy()
            for col in self.columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")

                df_encoded[col] = df_encoded[col].astype(str)


                for cat in self.categories_[col][1:]:  
                    new_col = f"{col}_{cat}"
                    df_encoded[new_col] = (df_encoded[col] == cat).astype(int)


                df_encoded.drop(columns=[col], inplace=True)

            return df_encoded
        except Exception as e:
            raise Exception(f"Error transforming with One-Hot Encoder: {e}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit + transform for training data
        """
        self.fit(df)
        return self.transform(df)

    def save(self, save_path: str):
        """
        Save the learned categories and metadata as a single artifact.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {
            "columns": list(self.columns),
            "categories": self.categories_
        }
        joblib.dump(payload, save_path)

    def load(self, load_path: str):
        """
        Load the saved one-hot encoder state.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"One-Hot Encoder bundle not found: {load_path}")

        payload = joblib.load(load_path)
        if "columns" not in payload or "categories" not in payload:
            raise ValueError("One-Hot Encoder bundle missing required fields")

        self.columns = tuple(payload["columns"])
        self.categories_ = payload["categories"]
