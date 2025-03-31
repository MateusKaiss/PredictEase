import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class MLModel:
    def __init__(self, model_type='linear'):
        model_map = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(),
            'xgb': XGBRegressor(),
            'lgbm': LGBMRegressor(),
        }

        if model_type not in model_map:
            raise ValueError(f'Model "{model_type}" is not supported.')

        self.model = model_map[model_type]
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X_future: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError('You must call `.fit()` before predicting.')

        return pd.Series(self.model.predict(X_future), index=X_future.index)
