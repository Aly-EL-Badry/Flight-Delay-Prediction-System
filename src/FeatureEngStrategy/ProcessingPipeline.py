from .EncodingStrategy.Label import LabelEncodingStrategy
from .EncodingStrategy.OneHot import OneHotEncodingStrategy
from .ScalingStrategy.CyclicScaler import CyclicEncodingStrategy
from .ScalingStrategy.StandardScaler import ScalingStrategy
from typing import List
import joblib
import os
import pandas as pd

class PreprocessingPipeline:
    def __init__(self, LabelCols: List[str], OheCols: List[str], CyclicCol: List[str], ScaleCols: List[str]):
        self.LabelCols = LabelCols
        self.OheCols = OheCols

        self.LabelEncoder = LabelEncodingStrategy(LabelCols)
        self.OheEncoder = OneHotEncodingStrategy(OheCols)
        self.Scaler = ScalingStrategy(ScaleCols)
        self.Cyclic = CyclicEncodingStrategy(CyclicCol)

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.LabelEncoder.apply(X)
        X = self.OheEncoder.apply(X)
        X = self.Scaler.apply(X)
        X = self.Cyclic.apply(X)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.LabelEncoder.transform(X)
        X = self.OheEncoder.transform(X)
        X = self.Scaler.transform(X)
        X = self.Cyclic.transform(X)
        return X
    
    def Save(self, savePath: str):
        """
        Save all encoders, scalers, cyclic encoders, and metadata as a single artifact.
        """
        os.makedirs(os.path.dirname(savePath) or ".", exist_ok=True)
        payload = {
            "LabelCols": self.LabelCols,
            "OheCols": self.OheCols,
            "CyclicCols": self.CyclicCols,
            "ScaleCols": self.ScaleCols,
            "LabelEncoder": self.LabelEncoder,
            "OheEncoder": self.OheEncoder,
            "Scaler": self.Scaler,
            "Cyclic": self.Cyclic
        }
        joblib.dump(payload, savePath)

    def Load(self, loadPath: str):
        """
        Load the saved pipeline state including encoders, scalers, and metadata.
        """
        if not os.path.exists(loadPath):
            raise FileNotFoundError(f"Pipeline file not found: {loadPath}")

        payload = joblib.load(loadPath)

        requiredKeys = ["LabelCols", "OheCols", "CyclicCols", "ScaleCols",
                        "LabelEncoder", "OheEncoder", "Scaler", "Cyclic"]
        for key in requiredKeys:
            if key not in payload:
                raise ValueError(f"Pipeline bundle missing required field: {key}")

        self.LabelCols = payload["LabelCols"]
        self.OheCols = payload["OheCols"]
        self.CyclicCols = payload["CyclicCols"]
        self.ScaleCols = payload["ScaleCols"]

        self.LabelEncoder = payload["LabelEncoder"]
        self.OheEncoder = payload["OheEncoder"]
        self.Scaler = payload["Scaler"]
        self.Cyclic = payload["Cyclic"]