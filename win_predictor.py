import numpy as np
import pandas as pd


class WinPredictor:
    def train(self, x: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> int:
        raise NotImplementedError

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
