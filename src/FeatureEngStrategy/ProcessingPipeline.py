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
            X[self.LabelCols] = self.LabelEncoder.apply(X[self.LabelCols])
        logger.info("finish Label encoder")

        logger.info("Start one hot encoder ")
        if self.OheCols:
            X = self.OheEncoder.apply(X)  # Should return one-pass transformed DataFrame
        logger.info("finish one hot encoder")

        logger.info("Start scale encoder")
        if self.ScaleCols:
            X[self.ScaleCols] = self.Scaler.apply(X[self.ScaleCols])
        logger.info("finish scale encoder")


        logger.info("start cyclic encoder")
        if self.CyclicCols:
            X = self.Cyclic.apply(X)
        logger.info("finish cyclic encoder")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Transform in-place where possible to avoid extra copies
        if self.LabelCols:
            X[self.LabelCols] = self.LabelEncoder.transform(X[self.LabelCols])
        
        if self.OheCols:
            X = self.OheEncoder.transform(X)

        if self.ScaleCols:
            X[self.ScaleCols] = self.Scaler.transform(X[self.ScaleCols].values)

        if self.CyclicCols:
            X[self.CyclicCols] = self.Cyclic.transform(X[self.CyclicCols])

        return X

    def Save(self, savePath: str):
        os.makedirs(os.path.dirname(savePath) or ".", exist_ok=True)
        joblib.dump(self.__dict__, savePath)

    def Load(self, loadPath: str):
        if not os.path.exists(loadPath):
            raise FileNotFoundError(f"Pipeline file not found: {loadPath}")
        payload = joblib.load(loadPath)
        self.__dict__.update(payload)
