import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predictease time series runner'
    )

    parser.add_argument(
        'endog_path', type=str, help='Path to the endogenous CSV file'
    )

    parser.add_argument(
        '--exog_path',
        type=str,
        default=None,
        help='Path to the exogenous CSV file (optional)',
    )

    parser.add_argument(
        '--explore', action='store_true', help='Initial data exploration plot'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['arima'],
        help='Model to train and predict (e.g., arima)',
    )

    parser.add_argument(
        '--forecast_steps',
        type=int,
        default=10,
        help='Number of steps ahead to forecast',
    )

    return parser.parse_args()
