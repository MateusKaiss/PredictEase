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


def load_and_prepare_data(endog_path, exog_path=None):
    print(f'Loading endogenous data from: {endog_path}')
    data: TimeSeriesDataset = load_data(endog_path, exog_path)

    df = data.endog.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    target_col = [col for col in df.columns if col != 'date'][0]
    y = df[target_col]

    return data, y


def explore_data(data: TimeSeriesDataset):
    plot_endogenous(data)
    plot_exogenous_over_time(data)
    plot_exog_vs_endog(data)


def run_model(
    model,
    data,
    y,
    forecast_steps,
    seasonal_length,
    window_size,
    epochs,
    batch_size,
):
    if model == 'arima':
        print(f'\n Training ARIMA model...')
        m = ARIMAModel()

    elif model == 'prophet':
        print(f'\n Training Prophet model...')
        m = ProphetModel()

    elif model in ['naive', 'mean', 'seasonal_naive']:
        print(f'\n Running baseline model: {model}')
        if model == 'naive':
            m = NaiveModel()
        elif model == 'mean':
            m = MeanModel()
        else:
            if seasonal_length is None:
                seasonal_length = 12
                print(
                    f' No seasonal_length provided. Using default: {seasonal_length}'
                )
            m = SeasonalNaiveModel(season_length=seasonal_length)

    elif model in ['linear', 'rf', 'xgb', 'lgbm']:
        if data.exog is None:
            raise ValueError(' Exogenous data is required for ML models.')

        print(f'\n Training ML model: {model}')
        exog = data.exog.copy()
        exog['date'] = pd.to_datetime(exog['date'])
        exog.set_index('date', inplace=True)
        X = exog.reindex(y.index).fillna(method='ffill')

        m = MLModel(model_type=model)
        m.fit(X, y)
        X_future = exog.iloc[-forecast_steps:]
        return m.predict(X_future)

    elif model == 'lstm':
        print(f'\n Training LSTM model...')
        m = LSTMModel(
            window_size=window_size, epochs=epochs, batch_size=batch_size
        )

    elif model == 'mlp':
        print(f'\n Training MLP model...')
        m = MLPModel(
            window_size=window_size, epochs=epochs, batch_size=batch_size
        )

    else:
        raise ValueError(f'Model "{model}" is not implemented.')

    m.fit(y)
    return m.predict(steps=forecast_steps)


def run(
    endog_path,
    exog_path=None,
    explore=False,
    model=None,
    forecast_steps=10,
    seasonal_length=12,
    window_size=12,
    epochs=100,
    batch_size=16,
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
