from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseEvaluator(ABC):
    """Abstract evaluator strategy for comparing ground-truth and predictions.

    Implementations must accept numpy arrays (or array-like) and return a
    serializable dictionary with metric names -> float values.
    """

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute metrics comparing y_true and y_pred."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Human-friendly name for the evaluator."""
        raise NotImplementedError

    def save(self, results: Dict[str, float], path: Optional[str] = None) -> None:
        """Optional: persist results to a path (JSON). Implementations can override.

        If `path` is None this is a no-op.
        """
        if path is None:
            return
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"evaluator": self.name(), "results": results}, f, indent=2)