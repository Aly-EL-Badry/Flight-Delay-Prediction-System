import numpy as np
from typing import Dict
from ..base import BaseEvaluator
from sklearn.metrics import r2_score


class R2Evaluator(BaseEvaluator):
    """Evaluator that returns the R^2 score and an "accuracy percent" = R2 * 100.

    Usage:
        ev = R2Evaluator()
        stats = ev.evaluate(y_true, y_pred)
    """

    def name(self) -> str:
        return "r2_evaluator"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        r2 = float(r2_score(y_true, y_pred))
        accuracy_percent = r2 * 100.0

        return {
            "r2": r2,
            "accuracy_percent": accuracy_percent,
        }