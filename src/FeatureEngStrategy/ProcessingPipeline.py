from .EncodingStrategy.Label import LabelEncodingStrategy
from .EncodingStrategy.OneHot import OneHotEncodingStrategy
from .ScalingStrategy.CyclicScaler import CyclicEncodingStrategy
from .ScalingStrategy.StandardScaler import ScalingStrategy
from typing import List
import joblib
import os
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PreprocessingPipeline:
    def __init__(self, LabelCols: List[str], OheCols: List[str], CyclicCol: List[str], ScaleCols: List[str]):
        self.LabelCols = LabelCols
        self.OheCols = OheCols
        self.CyclicCols = CyclicCol
        self.ScaleCols = ScaleCols

        self.LabelEncoder = LabelEncodingStrategy(LabelCols)
        self.OheEncoder = OneHotEncodingStrategy(OheCols)
        self.Scaler = ScalingStrategy(ScaleCols)
        self.Cyclic = CyclicEncodingStrategy(CyclicCol)

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Start Label encoder")
        if self.LabelCols:
            self.LabelEncoder.fit(X[self.LabelCols])
        logger.info("finish Label encoder")

        logger.info("Start one hot encoder ")
        if self.OheCols:
            self.OheEncoder.fit(X[self.OheCols])  # Should return one-pass transformed DataFrame
        logger.info("finish one hot encoder")

        logger.info("Start scale encoder")
        if self.ScaleCols:
            self.Scaler.fit(X[self.ScaleCols])
        logger.info("finish scale encoder")


        logger.info("start cyclic encoder")
        if self.CyclicCols:
            self.Cyclic.fit(X[self.CyclicCols])
        logger.info("finish cyclic encoder")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Transform in-place where possible to avoid extra copies
        print(X.shape)
        print("Transforming data using label encoding")
        if self.LabelCols:
            X[self.LabelCols] = self.LabelEncoder.transform(X[self.LabelCols])
            print("Shape after label encoding:", X.shape)
        
        print("Transforming data using one-hot encoding")
        if self.OheCols:
            X = self.OheEncoder.transform(X)
        print("Shape after label encoding:", X.shape)

        print("Transforming data using scaling")
        if self.ScaleCols:
            scaled_array = self.Scaler.transform(X[self.ScaleCols])
            X[self.ScaleCols] = pd.DataFrame(scaled_array, columns=self.ScaleCols, index=X.index)
        print("Shape after label encoding:", X.shape)

        print("Transforming data using cyclic encoding")
        if self.CyclicCols:
            X = self.Cyclic.transform(X)

        return X

    def Save(self, savePath: str):
        os.makedirs(os.path.dirname(savePath) or ".", exist_ok=True)
        joblib.dump(self.__dict__, savePath)

    def Load(self, loadPath: str):
        if not os.path.exists(loadPath):
            raise FileNotFoundError(f"Pipeline file not found: {loadPath}")
        payload = joblib.load(loadPath)
        self.__dict__.update(payload)
