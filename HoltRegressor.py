from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.utils.validation import check_array, check_is_fitted
import pandas as pd
import numpy as np

class HoltWintersTripleExponentialSmoothing(BaseEstimator, RegressorMixin):
    def __init__(self, trend=None, seasonal=None, seasonal_periods=None, damped=False):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped = damped

    def fit(self, X, y=None):
        # Convert to 1-D numpy array and apply a simple log transform for stability
        X = np.asarray(X).ravel()
        X = np.where(X <= 0, 1e-6, X)     
        self.X_mean_ = X.mean()
        self.X_std_ = X.std()
        X_norm = (np.log(X) - np.log(self.X_mean_)) / (self.X_std_ + 1e-6)

        self.model_ = ExponentialSmoothing(
            X_norm,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped=self.damped
        ).fit(optimized=True)
        return self

    def predict(self, X):
        steps = len(X) if hasattr(X, "__len__") else X
        preds = self.model_.forecast(steps)
        # Reverse normalization + exponentiate back
        return np.exp(preds * (self.X_std_ + 1e-6) + np.log(self.X_mean_))

    def score(self, X, y):
        preds = self.predict(X)
        return -np.mean((y - preds) ** 2)