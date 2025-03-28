from predictease.cli import parse_args
from predictease.core import run

if __name__ == '__main__':
    args = parse_args()
    run(args.endog_path, args.exog_path)
