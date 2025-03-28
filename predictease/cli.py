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

    return parser.parse_args()
