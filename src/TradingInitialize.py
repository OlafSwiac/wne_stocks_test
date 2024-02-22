import pandas as pd
import datetime
import OptionTradingFuntions as opt
from sklearn.exceptions import DataConversionWarning
import warnings
from ClassInitialize import TradingAlgorithmEnvironment

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def initialize_trading(stocks_symbols: list):
    periods = 144
    validation_portfolios = opt.get_validation_portfolio(stocks_symbols,
                                                         initial_time=datetime.datetime(2004, 1, 1),
                                                         periods=periods)

    initial_data = TradingAlgorithmEnvironment(stocks_symbols,
                                               initial_time=datetime.datetime(2004, 1, 1),
                                               stocks_owned={symbol: 0 for symbol in stocks_symbols},
                                               prediction_days=20,
                                               daily_cash=[100000],
                                               final_balance=0,
                                               stocks_owned_history=pd.DataFrame(columns=stocks_symbols),
                                               stocks_prices_history=pd.DataFrame(columns=stocks_symbols),
                                               daily_balances=[])
    for timedelta in range(0, periods):
        initial_data.update_timedelta(timedelta)
        if (timedelta + 1) % 3 == 0:
            initial_data.update_stocks()
        last_prices = dict(initial_data.stocks_prices_history.iloc[-1]) if initial_data.timedelta > 1 else 'DAY ONE'
        initial_data.update_last_prices(last_prices)
        initial_data.update_data()
        initial_data.simulate_multi_stock_trading()

    return initial_data, validation_portfolios
