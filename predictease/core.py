from typing import Optional, Tuple, Union

import pandas as pd

from .analysis.reader import TimeSeriesDataset, load_data
from .analysis.visualization import (
    plot_endogenous,
    plot_exog_vs_endog,
    plot_exogenous_over_time,
)
from .models.arima import ARIMAModel
from .models.baseline import MeanModel, NaiveModel, SeasonalNaiveModel
from .models.ml_models import MLModel
from .models.nn_models import LSTMModel, MLPModel
from .models.prophet import ProphetModel


def load_and_prepare_data(
    endog_path: str, exog_path: Optional[str] = None
) -> Tuple[TimeSeriesDataset, pd.Series]:
    print(f'Loading endogenous data from: {endog_path}')
    data: TimeSeriesDataset = load_data(endog_path, exog_path)

    df = data.endog.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    target_col = [col for col in df.columns if col != 'date'][0]
    y = df[target_col]

    return data, y


def explore_data(data: TimeSeriesDataset) -> None:
    plot_endogenous(data)
    plot_exogenous_over_time(data)
    plot_exog_vs_endog(data)


def get_model(
    model_name: str,
    data: TimeSeriesDataset,
    y: pd.Series,
    seasonal_length: Optional[int],
    window_size: int,
    epochs: int,
    batch_size: int,
) -> Union[
    ARIMAModel,
    ProphetModel,
    NaiveModel,
    MeanModel,
    SeasonalNaiveModel,
    LSTMModel,
    MLPModel,
    Tuple[MLModel, pd.DataFrame],
]:
    if model_name == 'arima':
        print('\n Training ARIMA model...')
        return ARIMAModel()

    elif model_name == 'prophet':
        print('\n Training Prophet model...')
        return ProphetModel()

    elif model_name == 'naive':
        print('\n Running baseline model: naive')
        return NaiveModel()

    elif model_name == 'mean':
        print('\n Running baseline model: mean')
        return MeanModel()

    elif model_name == 'seasonal_naive':
        print('\n Running baseline model: seasonal_naive')
        if seasonal_length is None:
            seasonal_length = 12
            print(
                f' No seasonal_length provided. Using default: {seasonal_length}'
            )
        return SeasonalNaiveModel(season_length=seasonal_length)

    elif model_name in ['linear', 'rf', 'xgb', 'lgbm']:
        if data.exog is None:
            raise ValueError(' Exogenous data is required for ML models.')

        print(f'\n Training ML model: {model_name}')
        exog = data.exog.copy()
        exog['date'] = pd.to_datetime(exog['date'])
        exog.set_index('date', inplace=True)
        X = exog.reindex(y.index).fillna(method='ffill')

        model = MLModel(model_type=model_name)
        model.fit(X, y)
        return model, exog

    elif model_name == 'lstm':
        print('\n Training LSTM model...')
        return LSTMModel(
            window_size=window_size, epochs=epochs, batch_size=batch_size
        )

    elif model_name == 'mlp':
        print('\n Training MLP model...')
        return MLPModel(
            window_size=window_size, epochs=epochs, batch_size=batch_size
        )

    else:
        raise ValueError(f'Model "{model_name}" is not implemented.')


def run_model(
    model: str,
    data: TimeSeriesDataset,
    y: pd.Series,
    forecast_steps: int,
    seasonal_length: Optional[int],
    window_size: int,
    epochs: int,
    batch_size: int,
):
    result = get_model(
        model, data, y, seasonal_length, window_size, epochs, batch_size
    )

    if isinstance(result, tuple):
        m, exog = result
        X_future = exog.iloc[-forecast_steps:]
        return m.predict(X_future)

    m = result
    m.fit(y)
    return m.predict(steps=forecast_steps)


def run(
    endog_path: str,
    exog_path: Optional[str] = None,
    explore: bool = False,
    model: Optional[str] = None,
    forecast_steps: int = 10,
    seasonal_length: Optional[int] = 12,
    window_size: int = 12,
    epochs: int = 100,
    batch_size: int = 16,
):
    data, y = load_and_prepare_data(endog_path, exog_path)

    if explore:
        explore_data(data)
    else:
        print('\n🛈 Skipping plots (use --explore to enable)')

    forecast = run_model(
        model=model,
        data=data,
        y=y,
        forecast_steps=forecast_steps,
        seasonal_length=seasonal_length,
        window_size=window_size,
        epochs=epochs,
        batch_size=batch_size,
    )

    print(f'\n Forecast:\n{forecast}')
