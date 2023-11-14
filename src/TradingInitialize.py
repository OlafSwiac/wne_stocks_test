import pandas as pd
import yfinance as yf
import datetime
import OptionTradingFuntions as opt
import matplotlib
import matplotlib.pyplot as plt
import TradingSimulator as trade


def initialize_trading(stocks_symbols: list):
    daily_balances = []
    final_balance = 0
    stocks_owned_history = pd.DataFrame(columns=stocks_symbols)
    daily_cash = [100000]
    stocks_owned = {symbol: 0 for symbol in stocks_symbols}
    for month in range(1, 3):
        start_date_train = datetime.datetime(2013, month * 3, 1)
        end_date_train = datetime.datetime(2017, month * 3, 1)

        start_date_test = datetime.datetime(2017, month * 3 - 1, 1)
        end_date_test = datetime.datetime(2017, month * 3, 1)

        start_date_predict = datetime.datetime(2017, month * 3, 1)
        end_date_predict = datetime.datetime(2017, month * 3 + 3, 1)

        df_decisions = pd.DataFrame()
        df_kelly = pd.DataFrame()

        # Initialize a dictionary to store stock data
        stocks_data = {}

        # Fetch data for each stock and apply feature engineering and model functions
        for stock_symbol in stocks_symbols:
            print(stock_symbol)
            data_train = yf.download(stock_symbol, start=start_date_train, end=end_date_train)
            data_test = yf.download(stock_symbol, start=start_date_test, end=end_date_test)
            data_predict = yf.download(stock_symbol, start=start_date_predict, end=end_date_predict)
            # data_with_lags = opt.create_lagged_features(data_train[['Close']].copy())
            # data_with_lags_test = opt.create_lagged_features(data_test[['Close']].copy())
            kelly_fractions, decisions = opt.get_predictions_and_kelly_criterion(data_train, data_test, data_predict,
                                                                                 15)
            stocks_data[stock_symbol] = data_train
            df_decisions[stock_symbol] = decisions
            df_kelly[stock_symbol] = kelly_fractions

        # Run the multi-stock trading simulation
        daily_balances_period, final_balance_period, stocks_owned_history_period, daily_cash_period = \
            trade.simulate_multi_stock_trading(stocks_symbols, stocks_data, df_decisions, df_kelly, stocks_owned,
                                               daily_cash[-1])

        daily_balances = daily_balances + daily_balances_period
        final_balance = final_balance_period
        stocks_owned_history = pd.concat([stocks_owned_history, stocks_owned_history_period]).reset_index(drop=True)
        daily_cash = daily_cash + daily_cash_period

    return daily_balances, final_balance, stocks_owned_history, daily_cash
