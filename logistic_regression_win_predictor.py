import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from win_feature_extractor import WinFeatureExtractor
from win_predictor import WinPredictor


class LogisticRegressionWinPredictor(WinPredictor):
    model: LogisticRegression
    feature_extractor: WinFeatureExtractor

    def __init__(self, feature_extractor: WinFeatureExtractor):
        self.model = LogisticRegression(penalty='none')
        self.feature_extractor = feature_extractor

    def train(self, x: pd.DataFrame, y: np.ndarray) -> None:
        features = self.feature_extractor.get_features(x)
        self.model.fit(features, y)

    def predict(self, x: pd.DataFrame) -> [int]:
        features = self.feature_extractor.get_features(x)
        return self.model.predict(features)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        features = self.feature_extractor.get_features(x)
        return self.model.predict_proba(features)
