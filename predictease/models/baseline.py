import pandas as pd


class NaiveModel:
    def fit(self, y: pd.Series):
        self.last_value = y.dropna().iloc[-1]

    def predict(self, steps: int = 1) -> pd.Series:
        return pd.Series([self.last_value] * steps)


class MeanModel:
    def fit(self, y: pd.Series):
        self.mean_value = y.mean()

    def predict(self, steps: int = 1) -> pd.Series:
        return pd.Series([self.mean_value] * steps)


class SeasonalNaiveModel:
    def __init__(self, season_length: int = 12):
        self.season_length = season_length

    def fit(self, y: pd.Series):
        y = y.dropna()
        if len(y) < self.season_length:
            raise ValueError('Series too short for seasonal naive.')
        self.season = y.iloc[-self.season_length :]

    def predict(self, steps: int = 1) -> pd.Series:
        repeats = -(-steps // len(self.season))
        forecast = list(self.season) * repeats
        return pd.Series(forecast[:steps])
