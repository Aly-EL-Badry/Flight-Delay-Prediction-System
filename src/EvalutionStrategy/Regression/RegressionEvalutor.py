import numpy as np
from typing import Dict
from ..base import BaseEvaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionEvaluator(BaseEvaluator):
    """Comprehensive regression evaluator computing MSE, MAE, R2 and accuracy percent.

    Returns a serializable dict with keys: mse, mae, r2, accuracy_percent
    """

    def name(self) -> str:
        return "regression_evaluator"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))


        return {
            "mse": mse,
            "mae": mae,
        }