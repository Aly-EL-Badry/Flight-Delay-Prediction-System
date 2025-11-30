from zenml import step
import numpy as np
from typing import Dict
from ..src.EvalutionStrategy.R2.R2Strategy import R2Evaluator
from ..src.EvalutionStrategy.Regression.RegressionEvalutor import RegressionEvaluator


@step
def evaluateModel(
    yPred: np.ndarray,
    yTest: np.ndarray
) -> Dict[str, float]:
    """
    Run both R2Evaluator and RegressionEvaluator and return merged metrics.

    Returned dict keys: r2, accuracy_percent, mse, mae (all float)
    """
    yTrueArr = np.asarray(yTest).ravel()
    yPredArr = np.asarray(yPred).ravel()

    r2_eval = R2Evaluator()
    r2_metrics = r2_eval.evaluate(yTrueArr, yPredArr)

    reg_eval = RegressionEvaluator()
    reg_metrics = reg_eval.evaluate(yTrueArr, yPredArr)

    merged = {}
    merged.update(r2_metrics)
    merged.update(reg_metrics)

    merged = {k: float(v) for k, v in merged.items()}

    return merged
