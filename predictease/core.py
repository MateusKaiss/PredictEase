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
from .models.prophet import ProphetModel


def run(
    endog_path,
    exog_path=None,
    explore=False,
    model=None,
    forecast_steps=10,
    seasonal_length=12,
):
    print(f'Loading endogenous data from: {endog_path}')
    data: TimeSeriesDataset = load_data(endog_path, exog_path)

    if explore:
        plot_endogenous(data)
        plot_exogenous_over_time(data)
        plot_exog_vs_endog(data)
    else:
        print('\n🛈 Skipping plots (use --explore to enable)')

    df = data.endog.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    target_col = [col for col in df.columns if col != 'date'][0]
    y = df[target_col]

    if model == 'arima':
        print(
            f'\n Training ARIMA model and forecasting {forecast_steps} steps...'
        )
        arima_model = ARIMAModel()
        arima_model.fit(y)
        forecast = arima_model.predict(steps=forecast_steps)
        print(f'\n Forecast:\n{forecast}')

    elif model == 'prophet':
        print(
            f'\n Training Prophet model and forecasting {forecast_steps} steps...'
        )
        prophet_model = ProphetModel()
        prophet_model.fit(y)
        forecast = prophet_model.predict(steps=forecast_steps)
        print(f'\n Forecast:\n{forecast}')

    elif model in ['linear', 'rf', 'xgb', 'lgbm']:
        if data.exog is None:
            raise ValueError(' Exogenous data is required for ML models.')

        print(
            f'\n Training ML model ({model}) and forecasting {forecast_steps} steps...'
        )

        exog = data.exog.copy()
        exog['date'] = pd.to_datetime(exog['date'])
        exog.set_index('date', inplace=True)

        X = exog.reindex(y.index)

        if X.isnull().any().any():
            print(
                ' Exogenous data has missing values. Filling them with forward fill.'
            )
            X = X.fillna(method='ffill')

        ml_model = MLModel(model_type=model)
        ml_model.fit(X, y)

        X_future = exog.iloc[-forecast_steps:]
        forecast = ml_model.predict(X_future)

        print(f'\n Forecast:\n{forecast}')

    elif model in ['naive', 'mean', 'seasonal_naive']:
        print(f'\n Running baseline model: {model}...')

        if model == 'naive':
            baseline_model = NaiveModel()
        elif model == 'mean':
            baseline_model = MeanModel()
        elif model == 'seasonal_naive':
            baseline_model = SeasonalNaiveModel(season_length=seasonal_length)

        baseline_model.fit(y)
        forecast = baseline_model.predict(steps=forecast_steps)

        print(f'\n Forecast:\n{forecast}')

    elif model:
        print(f' Model "{model}" is not yet implemented.')
