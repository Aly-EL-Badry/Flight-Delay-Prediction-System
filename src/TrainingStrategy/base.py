# src/your_package/models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

class BaseModel(ABC):
    """
    Abstract interface for trainable models.

    Implementations must not hard-code hyperparameters or shapes.
    All hyperparameters and shapes must be provided via constructor or method args.
    """

    @abstractmethod
    def build(self, input_shape: int, **kwargs) -> None:
        """Build the model architecture. input_shape = number of features."""
        raise NotImplementedError

    @abstractmethod
    def compile(self, compile_kwargs: Dict[str, Any]) -> None:
        """Compile the model with optimizer/loss/metrics."""
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **fit_kwargs
    ) -> Any:
        """Train the model. Returns backend-specific history object."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray, **predict_kwargs) -> np.ndarray:
        """Return predictions for x."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist model artifact and optional metadata to path (atomic)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model artifact from path."""
        raise NotImplementedError
