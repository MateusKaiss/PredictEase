import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predictease time series runner'
    )

    parser.add_argument(
        'endog_path',
        type=str,
        help='Path to the endogenous CSV file',
    )

    parser.add_argument(
        '--exog_path',
        type=str,
        default=None,
        help='Path to the exogenous CSV file (optional)',
    )

    parser.add_argument(
        '--explore',
        action='store_true',
        help='Initial data exploration plot',
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=[
            'arima',
            'prophet',
            'linear',
            'rf',
            'xgb',
            'lgbm',
            'naive',
            'mean',
            'seasonal_naive',
            'lstm',
            'mlp',
        ],
        help=(
            'Model to train and predict. '
            'Options: arima, prophet, linear, rf, xgb, lgbm, '
            'naive, mean, seasonal_naive, lstm, mlp'
        ),
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=12,
        help='Window size for neural network models (e.g., LSTM, MLP)',
    )

    parser.add_argument(
        '--hidden_units',
        type=int,
        default=64,
        help='Number of hidden units in LSTM/MLP layers',
    )

    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'tanh', 'sigmoid', 'linear'],
        help='Activation function for neural networks',
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for neural network models',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training neural network models',
    )

    parser.add_argument(
        '--forecast_steps',
        type=int,
        default=10,
        help='Number of steps ahead to forecast',
    )

    parser.add_argument(
        '--seasonal_length',
        type=int,
        default=None,
        help='Season length for seasonal naive model (e.g., 12 for monthly data with yearly seasonality)',
    )

    return parser.parse_args()
