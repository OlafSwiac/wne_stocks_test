import pandas as pd
import datetime
import trading_functions as opt
# from sklearn.exceptions import DataConversionWarning
import warnings
from trading_environment import TradingAlgorithmEnvironment
import simplejson

pd.set_option('mode.chained_assignment', None)
# This code will not complain!

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def initialize_trading(parameters: dict, generate_lists_of_stocks: bool):
    periods = 210

    price_bought = {}
    for stock in parameters['stocks']:
        price_bought[stock] = 0

    initial_data = TradingAlgorithmEnvironment(parameters['stocks'],
                                               initial_time=datetime.datetime(2005, 1, 1),
                                               stocks_owned={symbol: 0 for symbol in parameters['stocks']},
                                               prediction_days=20,
                                               daily_cash=[100000],
                                               final_balance=0,
                                               transaction_cost=parameters['tc'],
                                               price_bought=price_bought,
                                               stocks_owned_history=pd.DataFrame(columns=parameters['stocks']),
                                               stocks_prices_history=pd.DataFrame(columns=parameters['stocks']),
                                               daily_balances=[],
                                               stocks_file=parameters['metrics'],
                                               short_stocks_file='Stock_lists/new_stocks_test_15_second_best_ASD_2005_train.json',
                                               stop_loss_long=parameters['sl_long'],
                                               stop_loss_short=parameters['sl_short'],
                                               what_kelly=parameters['kelly'],
                                               volatility_gate=parameters['vg'],
                                               generate_lists_of_stocks=generate_lists_of_stocks)

    stocks_lists = {}
    for timedelta in range(0, periods):
        initial_data.update_timedelta(timedelta)
        "print(initial_data.is_short_stock)"
        "last_prices = dict(initial_data.stocks_prices_history.iloc[-1]) if initial_data.timedelta > 1 else 'DAY ONE'"
        last_prices = 'DAY ONE'
        if ~generate_lists_of_stocks:
            if ((timedelta + 1) % parameters['change'] == 0) & (timedelta > 0):  # zmiana na comiesieczne zmiany
                initial_data.update_stocks(timedelta=timedelta)
                last_prices = 'DAY ONE'
        if generate_lists_of_stocks:
            if timedelta > -1:
                initial_data.update_stocks(timedelta=timedelta)
        stocks_lists[timedelta] = initial_data.stocks_symbols
        initial_data.update_last_prices(last_prices)
        initial_data.update_data()
        if ~generate_lists_of_stocks:
            initial_data.simulate_multi_stock_trading()

    "print(stocks_lists)"
    if generate_lists_of_stocks:
        with open("Stock_lists/new_stocks_test_15_second_best_ASD_2005_train.json", "w") as fp:
            simplejson.dump(stocks_lists, fp)

    return initial_data
