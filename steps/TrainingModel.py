import logging 
from src.TrainingStrategy.NeuralNetwork.NN import KerasRegressor

from zenml import step
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@step
def trainModel(
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    modelParams: Dict[str, Any],
    compileParams: Dict[str, Any],
    fitParams: Dict[str, Any]
) -> KerasRegressor:
    """
    Train a KerasRegressor model using only training data (no validation).
    """

    model = KerasRegressor(**modelParams)

    model.build(input_shape=(xTrain.shape[1],))

    model.compile(compileParams)

    model.fit(xTrain, yTrain, **fitParams)

    return model

@step
def predict(model: KerasRegressor, xTest: np.ndarray) -> np.ndarray:
    preds = model.predict(xTest)
    return preds

@step
def saveModel(model: KerasRegressor, savePath: str) -> str:
    model.save(savePath)
