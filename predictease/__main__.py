from predictease.cli import parse_args
from predictease.core import run

if __name__ == '__main__':
    args = parse_args()
    run(
        args.endog_path,
        args.exog_path,
        explore=args.explore,
        model=args.model,
        forecast_steps=args.forecast_steps,
        seasonal_length=args.seasonal_length,
        window_size=args.window_size,
        epochs=args.epochs,
    )

