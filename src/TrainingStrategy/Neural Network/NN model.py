
import os
import json
import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple
from keras import layers
from ..base import BaseModel

class KerasRegressor(BaseModel):
    def __init__(self, **hyperparams):
        """
        hyperparams may include:
        - hidden_units: list[int]
        - dropout_rates: list[float]
        - activation: str
        - learning_rate: float
        - batch_norm: bool
        """
        self.hyperparams = hyperparams
        self.model = None

    def build(self, input_shape: int, **kwargs) -> None:
        hp = self.hyperparams

        hidden_units = hp.get("hidden_units", [256, 128, 64])
        dropout_rates = hp.get("dropout_rates", [0.0, 0.05, 0.05])
        activation = hp.get("activation", "swish")
        use_bn = hp.get("batch_norm", True)

        layers_list = [layers.Input(shape=(input_shape,))]

        for units, dr in zip(hidden_units, dropout_rates):
            layers_list.append(layers.Dense(units))

            if use_bn:
                layers_list.append(layers.BatchNormalization())

            layers_list.append(layers.Activation(activation))

            if dr > 0:
                layers_list.append(layers.Dropout(dr))

        layers_list.append(layers.Dense(1))  # regression output

        self.model = keras.Sequential(layers_list)

    def compile(self, compile_kwargs: Dict[str, Any]) -> None:
        """
        Compile the model with optimizer/loss/metrics.

        Parameters
        ----------
        compile_kwargs : Dict[str, Any]
            A dictionary containing the compile parameters.

        Returns
        -------
        None
        """
        self.model.compile(**compile_kwargs)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **fit_kwargs
    ) -> Any:
        return self.model.fit(
            x,
            y,
            validation_data=validation_data,
            **fit_kwargs
        )

    def predict(self, x: np.ndarray, **predict_kwargs) -> np.ndarray:
        return self.model.predict(x, **predict_kwargs)

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(path, exist_ok=True)

        # save model
        self.model.save(os.path.join(path, "model.keras"))

        # save metadata + hyperparams
        meta = {"hyperparams": self.hyperparams}
        if metadata:
            meta["metadata"] = metadata

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        self.model = keras.models.load_model(os.path.join(path, "model.keras"))

        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.hyperparams = json.load(f)["hyperparams"]
