import logging
from xml.parsers.expat import model 
from src.TrainingStrategy.NeuralNetwork.NN import KerasRegressor
from keras.optimizers import Adam, SGD, RMSprop

from zenml import step
import numpy as np
import pandas as pd
from typing import Any, Dict

import os

MODEL_DIR = os.path.abspath("artifacts\models")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_optimizer(opt_dict):
    """
    Convert dictionary format into a Keras optimizer instance.
    """
    if isinstance(opt_dict, dict):
        class_name = opt_dict.get("class_name")
        config = opt_dict.get("config", {})
        # map class_name string to actual class
        optimizers_map = {
            "Adam": Adam,
            "SGD": SGD,
            "RMSprop": RMSprop
        }
        if class_name not in optimizers_map:
            raise ValueError(f"Unknown optimizer {class_name}")
        return optimizers_map[class_name](**config)
    return opt_dict  # already a Keras optimizer instance


@step
def trainModel(
    xTrain: pd.DataFrame,
    yTrain: pd.Series,
    modelParams: Dict[str, Any],
    compileParams: Dict[str, Any],
    fitParams: Dict[str, Any]
):
    """
    Train a KerasRegressor model using only training data (no validation).
    """

    model = KerasRegressor(**modelParams)

    model.build(input_shape=(xTrain.shape[1]))

    optimizer_instance = get_optimizer(compileParams["optimizer"])
    model.compile({
        "optimizer": optimizer_instance,
        "loss": compileParams["loss"],
        "metrics": compileParams.get("metrics", [])
    })

    model.fit(xTrain, yTrain, **fitParams)

    model.save(MODEL_DIR)
    return True

@step
def predict(xTest: pd.DataFrame, done: bool) -> pd.Series:

    model_path = os.path.join(MODEL_DIR, "model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = KerasRegressor()
    model.load(MODEL_DIR)

    preds = model.predict(xTest.values)
    return pd.Series(preds.squeeze()) 