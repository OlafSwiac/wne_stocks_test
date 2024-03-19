import numpy as np
# import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import trading_initialize as trade
import datetime
from trading_functions import get_max_dropdown, make_date_column

matplotlib.use('TkAgg')
# yf.pdr_override()

matplotlib.pyplot.yscale('log')

stock_dict = {'sharpe': ['PPL', 'NEE', 'BR', 'MKC', 'NTRS', 'KO', 'YUM', 'SCHW', 'UNP', 'TXT', 'COST', 'STT',
                         'CLX', 'OXY', 'APA', 'MO', 'MRK', 'AAPL', 'PEG', 'THC', 'MSFT', 'CL', 'FE', 'D', 'DE',
                         'EOG', 'WAT', 'CF', 'HES', 'ADM']}

"""'return': ['CL', 'BK', 'AFL', 'D', 'EP', 'PEG', 'MRK', 'FE', 'MCK', 'YUM', 'X', 'COST', 'UNP',
           'NTRS', 'TXT', 'SCHW', 'GPS', 'BR', 'WAT', 'STT', 'EOG', 'OXY', 'APA', 'MSFT', 'UNH',
           'AAPL', 'DE', 'ADM', 'THC', 'HES']"""

start_day = datetime.datetime(2012, 1, 1)
end_day = datetime.datetime(2021, 1, 1)
dji = pd.read_csv('Stock_data_all_sp500/^DJI_data.csv')
dji['Date'] = pd.to_datetime(dji['Date'])
dates = dji['Date']

for key, stocks in stock_dict.items():
    Results, validation_portfolios = trade.initialize_trading(stocks)

    daily_balances = make_date_column(Results.daily_balances, dates)
    val_0 = make_date_column(validation_portfolios[0], dates)
    val_1 = make_date_column(validation_portfolios[1], dates)

    dji = dji.loc[0:len(daily_balances)]
    dji.set_index('Date', inplace=True)

    plt.plot(daily_balances)
    plt.plot(val_0)
    plt.plot(val_1)
    plt.plot(dji['Adj Close'] * 100000 / dji.iloc[0]['Adj Close'])
    plt.legend(['Predictions', 'Dow Jones Industrial Average'])
    plt.title(f'Portfolio value with stocks picked by {key}')
    plt.show()

    print(f'Max dropdown for predictions: {get_max_dropdown(Results.daily_balances)}')
    print(f'Max dropdown for validation portfolio 0: {get_max_dropdown(validation_portfolios[0])}')
    print(f'Max dropdown for validation portfolio 1: {get_max_dropdown(validation_portfolios[1])}')

    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')

    returns_list_v0 = np.array(validation_portfolios[0], dtype=float)
    returns_list_v0 = np.diff(returns_list_v0) / returns_list_v0[:-1]
    print(
        f'Sharpe ratio on validation portfolio 0: {np.mean(returns_list_v0) * np.sqrt(252) / np.std(returns_list_v0)}')
    returns_list_v0 = list(returns_list_v0)
    returns_list_v0_2 = np.array([i if i < 0 else 0 for i in returns_list_v0])
    print(
        f'Sortino ratio on validation portfolio 0: {np.mean(returns_list_v0) * np.sqrt(252) / np.std(returns_list_v0_2)}')

    returns_list_v1 = np.array(validation_portfolios[1], dtype=float)
    returns_list_v1 = np.diff(returns_list_v1) / returns_list_v1[:-1]
    print(
        f'Sharpe ratio on validation portfolio 1: {np.mean(returns_list_v1) * np.sqrt(252) / np.std(returns_list_v1)}')
    returns_list_v1 = list(returns_list_v1)
    returns_list_v1_2 = np.array([i if i < 0 else 0 for i in returns_list_v1])
    print(
        f'Sortino ratio on validation portfolio 0: {np.mean(returns_list_v1) * np.sqrt(252) / np.std(returns_list_v1_2)}')
