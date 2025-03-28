from .analysis.reader import TimeSeriesDataset, load_data


def run(endog_path, exog_path=None):
    print(f'Loading endogenous data from: {endog_path}')
    data = load_data(endog_path, exog_path)
