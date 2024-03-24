import pandas as pd
import datetime
import trading_functions as opt
# from sklearn.exceptions import DataConversionWarning
import warnings
from trading_environment import TradingAlgorithmEnvironment
import simplejson

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def initialize_trading(stocks_symbols: list):
    periods = 96

    price_bought = {}
    for stock in stocks_symbols:
        price_bought[stock] = 0

    initial_data = TradingAlgorithmEnvironment(stocks_symbols,
                                               initial_time=datetime.datetime(2009, 1, 1),
                                               stocks_owned={symbol: 0 for symbol in stocks_symbols},
                                               prediction_days=20,
                                               daily_cash=[100000],
                                               final_balance=0,
                                               price_bought=price_bought,
                                               stocks_owned_history=pd.DataFrame(columns=stocks_symbols),
                                               stocks_prices_history=pd.DataFrame(columns=stocks_symbols),
                                               daily_balances=[],
                                               stocks_file='Stock_lists/new_stocks_test_20_2006.json')

    validation_portfolios = opt.get_validation_portfolios(stocks_start=stocks_symbols,
                                                          stock_lists=initial_data.stocks_lists_for_each_change,
                                                          periods=periods,
                                                          initial_date=datetime.datetime(2009, 1, 1))

    stocks_lists = {}
    for timedelta in range(0, periods):
        initial_data.update_timedelta(timedelta)
        last_prices = dict(initial_data.stocks_prices_history.iloc[-1]) if initial_data.timedelta > 1 else 'DAY ONE'
        "last_prices = 'DAY ONE'"
        if (timedelta + 1) % 3 == 0: # zmiana na comiesieczne zmiany
            initial_data.update_stocks(timedelta=timedelta)
            last_prices = 'DAY ONE'
        stocks_lists[timedelta] = initial_data.stocks_symbols
        initial_data.update_last_prices(last_prices)
        initial_data.update_data()
        initial_data.simulate_multi_stock_trading()

    "print(stocks_lists)"

    """with open("Stock_lists/new_stocks_test_20_2006_2.json", "w") as fp:
        simplejson.dump(stocks_lists, fp)"""

    return initial_data, validation_portfolios
