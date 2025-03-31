import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA


class ARIMAModel:
    def __init__(self, seasonal=False, m=1, max_order=5):
        self.seasonal = seasonal
        self.m = m
        self.max_order = max_order
        self.model = None
        self.fitted_model = None
        self.order = None

    def fit(self, y: pd.Series):
        auto_model = auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_order=self.max_order,
        )
        self.order = auto_model.order
        print(f' Auto ARIMA selected order: {self.order}')

        self.model = StatsmodelsARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()

    def predict(self, steps: int = 1) -> pd.Series:
        if self.fitted_model is None:
            raise RuntimeError('You need to call `.fit()` before predicting.')
        return self.fitted_model.forecast(steps=steps)
