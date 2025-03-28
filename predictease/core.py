def run(endog_path, exog_path=None):
    print(f'Loading endogenous data from: {endog_path}')

    if exog_path:
        print(f'\nLoading exogenous data from: {exog_path}')
        print('\nExogenous data preview:')
    else:
        exog = None
        print('\nNo exogenous data provided.')
