import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TimeSeriesDataset:
    endog: pd.DataFrame
    exog: Optional[pd.DataFrame] = None


def load_data(
    endog_path: str, exog_path: Optional[str] = None
) -> TimeSeriesDataset:
    if not os.path.exists(endog_path):
        raise FileNotFoundError(f'Endogenous file not found: {endog_path}')

    endog = pd.read_csv(endog_path)

    if exog_path:
        if not os.path.exists(exog_path):
            raise FileNotFoundError(f'Exogenous file not found: {exog_path}')
        exog = pd.read_csv(exog_path)
    else:
        exog = None

    return TimeSeriesDataset(endog=endog, exog=exog)
