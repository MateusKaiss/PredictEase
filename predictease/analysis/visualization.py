import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.reader import TimeSeriesDataset


def plot_endogenous(data: TimeSeriesDataset):
    df = data.endog.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    endog_col = [col for col in data.endog.columns if col != 'date'][0]
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[endog_col], marker='o', linestyle='-')
    plt.title(f'{endog_col} over time')
    plt.xlabel('Date')
    plt.ylabel(endog_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_exog_vs_endog(data: TimeSeriesDataset):
    if data.exog is None:
        print('No exogenous data to plot.')
        return

    exog = data.exog.copy()
    endog = data.endog.copy()

    exog['date'] = pd.to_datetime(exog['date'])
    endog['date'] = pd.to_datetime(endog['date'])

    merged = pd.merge(endog, exog, on='date')
    endog_col = [col for col in data.endog.columns if col != 'date'][0]

    for col in exog.columns:
        if col != 'date':
            plt.figure(figsize=(6, 4))
            plt.scatter(merged[col], merged['sales'])
            plt.title(f'{endog_col} vs {col}')
            plt.xlabel(col)
            plt.ylabel(f'{endog_col}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_exogenous_over_time(data: TimeSeriesDataset):
    if data.exog is None:
        print('No exogenous data to plot.')
        return

    exog = data.exog.copy()
    exog['date'] = pd.to_datetime(exog['date'])
    exog.set_index('date', inplace=True)

    for col in exog.columns:
        if col != 'date':
            plt.figure(figsize=(8, 4))
            plt.plot(exog.index, exog[col], marker='o', linestyle='-')
            plt.title(f'{col} over time')
            plt.xlabel('Date')
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
