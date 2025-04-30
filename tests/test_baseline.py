import pandas as pd
import pytest

from predictease.models.baseline import (
    MeanModel,
    NaiveModel,
    SeasonalNaiveModel,
)

SERIE = pd.Series([1, 2, 3, 4, 5])


def test_naive_model():
    model = NaiveModel()
    model.fit(SERIE)
    forecast = model.predict(steps=3)

    expected = pd.Series([5, 5, 5])
    pd.testing.assert_series_equal(forecast.reset_index(drop=True), expected)


def test_mean_model():
    model = MeanModel()
    model.fit(SERIE)
    forecast = model.predict(steps=3)

    expected_mean = SERIE.mean()
    expected = pd.Series([expected_mean] * 3)
    pd.testing.assert_series_equal(forecast.reset_index(drop=True), expected)


def test_seasonal_naive_model():
    model = SeasonalNaiveModel(season_length=3)
    model.fit(SERIE)
    forecast = model.predict(steps=5)

    expected = pd.Series([3, 4, 5, 3, 4])
    pd.testing.assert_series_equal(forecast.reset_index(drop=True), expected)


def test_seasonal_naive_error():
    with pytest.raises(ValueError):
        SeasonalNaiveModel(season_length=10).fit(pd.Series([1, 2, 3]))
