import pandas as pd

from .analysis.reader import TimeSeriesDataset, load_data
from .analysis.visualization import (
    plot_endogenous,
    plot_exog_vs_endog,
    plot_exogenous_over_time,
)
from .models.arima import ARIMAModel


def run(
    endog_path, exog_path=None, explore=False, model=None, forecast_steps=10
):
    print(f'Loading endogenous data from: {endog_path}')
    data: TimeSeriesDataset = load_data(endog_path, exog_path)

    if explore:
        plot_endogenous(data)
        plot_exogenous_over_time(data)
        plot_exog_vs_endog(data)
    else:
        print('\n🛈 Skipping plots (use --explore to enable)')

    if model == 'arima':
        print(
            f'\ Training ARIMA model and forecasting {forecast_steps} steps...'
        )
        df = data.endog.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        target_col = [col for col in df.columns if col != 'date'][0]
        y = df[target_col]

        arima_model = ARIMAModel()
        arima_model.fit(y)
        forecast = arima_model.predict(steps=forecast_steps)

        print(f'\n Forecast:\n{forecast}')
    elif model:
        print(f' Model "{model}" is not yet implemented.')
