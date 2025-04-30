import re

import numpy as np
import pandas as pd
import pytest

from predictease.models.ml_models import MLModel


def generate_linear_exog_and_target(forecast_steps=5):
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    df_exog = pd.DataFrame(
        {
            'date': dates,
            'feature1': np.arange(60),
            'feature2': np.sin(np.arange(60) / 2),
        }
    )

    y = 3 * df_exog['feature1'] + 2 * df_exog['feature2'] + 5
    y += np.random.normal(0, 0.1, size=len(y))

    df_endog = pd.DataFrame(
        {
            'date': dates,
            'y': y,
        }
    )

    future_dates = pd.date_range(
        dates[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'
    )
    future_exog = pd.DataFrame(
        {
            'date': future_dates,
            'feature1': np.arange(60, 60 + forecast_steps),
            'feature2': np.sin(np.arange(60, 60 + forecast_steps) / 2),
        }
    )

    # Aqui está o que estava faltando:
    y_expected = 3 * future_exog['feature1'] + 2 * future_exog['feature2'] + 5

    return (
        df_exog.drop(columns=['date']),
        y,
        future_exog.drop(columns=['date']),
        y_expected,
    )


def test_mlmodel_linear_forecast_expected():
    X, y, X_future, y_expected = generate_linear_exog_and_target()

    model = MLModel('linear')
    model.fit(X, y)
    forecast = model.predict(X_future)

    assert isinstance(forecast, pd.Series)
    assert len(forecast) == len(X_future)

    mae = np.mean(np.abs(forecast.values - y_expected.values))
    print(
        'Forecast vs Expected:\n',
        pd.DataFrame(
            {'forecast': forecast.values, 'expected': y_expected.values}
        ),
    )
    print(f'MAE: {mae:.4f}')

    assert mae < 1, f'Erro absoluto médio alto: {mae:.4f}'


def test_mlmodel_linear_predict_before_fit():
    model = MLModel('linear')
    X_future = pd.DataFrame({'feature1': [1], 'feature2': [0.84]})

    with pytest.raises(
        RuntimeError,
        match=re.escape('You must call `.fit()` before predicting.'),
    ):
        model.predict(X_future)
