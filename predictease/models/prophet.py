import pandas as pd
from prophet import Prophet


class ProphetModel:
    def __init__(self):
        self.model = Prophet()
        self.fitted_model = None

    def fit(self, y: pd.Series):
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError('You need to call `.fit()` before predicting.')

        df = y.reset_index()
        df.columns = ['ds', 'y']

        self.model.fit(df)
        self.fitted_model = self.model

    def predict(self, steps: int = 1) -> pd.DataFrame:
        if self.fitted_model is None:
            raise RuntimeError('You need to call `.fit()` before predicting.')

        last_date = self.fitted_model.history['ds'].max()
        freq = pd.infer_freq(self.fitted_model.history['ds'])
        if freq is None:
            freq = 'D'

        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
