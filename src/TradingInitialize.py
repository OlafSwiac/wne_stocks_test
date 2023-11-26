import pandas as pd
import yfinance as yf
import datetime
import OptionTradingFuntions as opt
import matplotlib
import matplotlib.pyplot as plt
import TradingSimulator as trade
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def initialize_trading(stocks_symbols: list):
    daily_balances = []
    final_balance = 0
    stocks_owned_history = pd.DataFrame(columns=stocks_symbols)
    stocks_prices_history = pd.DataFrame(columns=stocks_symbols)
    daily_cash = [100000]
    stocks_owned = {symbol: 0 for symbol in stocks_symbols}
    initial_time = datetime.datetime(2012, 1, 1)
    for timedelta in range(0, 10):
        start_date_train = initial_time + datetime.timedelta(days=2 * 30 * timedelta)
        end_date_train = start_date_train + datetime.timedelta(days=4 * 365)

        start_date_test = start_date_train + datetime.timedelta(days=4 * 365 - 45)
        end_date_test = end_date_train

        start_date_predict = end_date_train
        end_date_predict = start_date_predict + datetime.timedelta(days=2 * 30)

        df_decisions = pd.DataFrame()
        df_kelly = pd.DataFrame()

        # Initialize a dictionary to store stock data
        stocks_data = {}

        # Fetch data for each stock and apply feature engineering and model functions
        for stock_symbol in stocks_symbols:
            # print(stock_symbol)
            """data_train = yf.download(stock_symbol, start=start_date_train, end=end_date_train)
            data_test = yf.download(stock_symbol, start=start_date_test, end=end_date_test)
            data_predict = yf.download(stock_symbol, start=start_date_predict, end=end_date_predict)"""

            data_train = pd.read_csv(f'Stock_Data/{stock_symbol}_data.csv')
            data_train = data_train[
                (data_train['Date'] >= str(start_date_train)[0:11]) & (data_train['Date'] <= str(end_date_train)[0:11])]

            data_test = pd.read_csv(f'Stock_Data/{stock_symbol}_data.csv')
            data_test = data_test[
                (data_test['Date'] >= str(start_date_test)[0:11]) & (data_test['Date'] <= str(end_date_test)[0:11])]

            data_predict = pd.read_csv(f'Stock_Data/{stock_symbol}_data.csv')
            data_predict = data_predict[(data_predict['Date'] >= str(start_date_predict)[0:11]) & (
                        data_predict['Date'] <= str(end_date_predict)[0:11])]

            kelly_fractions, decisions = opt.get_predictions_and_kelly_criterion(data_train, data_test, data_predict,
                                                                                 20)
            stocks_data[stock_symbol] = data_train
            df_decisions[stock_symbol] = decisions
            df_kelly[stock_symbol] = kelly_fractions

        # Run the multi-stock trading simulation
        daily_balances_period, final_balance_period, stocks_owned_history_period, stocks_prices_history_period, daily_cash_period = \
            trade.simulate_multi_stock_trading(stocks_symbols, stocks_data, df_decisions, df_kelly, stocks_owned,
                                               daily_cash[-1])

        daily_balances = daily_balances + daily_balances_period
        final_balance = final_balance_period
        stocks_owned_history = pd.concat([stocks_owned_history, stocks_owned_history_period]).reset_index(drop=True)
        stocks_prices_history = pd.concat([stocks_prices_history, stocks_prices_history_period]).reset_index(drop=True)
        daily_cash = daily_cash + daily_cash_period

    return daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash
