from .analysis.reader import TimeSeriesDataset, load_data
from .analysis.visualization import (
    plot_endogenous,
    plot_exog_vs_endog,
    plot_exogenous_over_time,
)


def run(endog_path, exog_path=None, explore=False):
    print(f'Loading endogenous data from: {endog_path}')
    data: TimeSeriesDataset = load_data(endog_path, exog_path)

    if explore:
        plot_endogenous(data)
        plot_exogenous_over_time(data)
        plot_exog_vs_endog(data)
    else:
        print('\n🛈 Skipping plots (use --explore to enable)')
