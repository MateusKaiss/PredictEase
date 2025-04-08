import numpy as np
import pandas as pd

from predictease.models.arima import ARIMAModel


def generate_linear_series(n=50, noise_std=0.25, seed=42):
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std, n)
    return pd.Series([2 * x + 5 for x in range(n)]) + noise


def test_arima_fit_and_forecast_linear():
    y = generate_linear_series(50)
    model = ARIMAModel(seasonal=False, max_order=5)
    model.fit(y)

    assert isinstance(model.order, tuple)
    assert (
        model.order[1] >= 1
    ), 'Expected at least one order of difference (trend)'

    forecast = model.predict(steps=5)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == 5

    last = y.iloc[-1]
    expected = pd.Series([last + 2 * (i + 1) for i in range(5)])
    mae = np.mean(np.abs(forecast.values - expected.values))
    assert mae < 1, f'High mean absolute error: {mae}'
