import numpy as np
# import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import trading_initialize as trade
import datetime
from trading_functions import get_max_drawdown, make_date_column
import metrics as m

matplotlib.use('TkAgg')
# yf.pdr_override()

matplotlib.pyplot.yscale('log')

stock_dict = {'stocks': ["PLD", "FCX", "CAT", "DOV", "LEG", "X", "ASH", "SRE", "LH", "AAPL", "WY", "XEL", "GILD", "SLB", "DRI"]}

"""'return': ['CL', 'BK', 'AFL', 'D', 'EP', 'PEG', 'MRK', 'FE', 'MCK', 'YUM', 'X', 'COST', 'UNP',
           'NTRS', 'TXT', 'SCHW', 'GPS', 'BR', 'WAT', 'STT', 'EOG', 'OXY', 'APA', 'MSFT', 'UNH',
           'AAPL', 'DE', 'ADM', 'THC', 'HES']"""
"""'sharpe': ['PPL', 'NEE', 'BR', 'MKC', 'NTRS', 'KO', 'YUM', 'SCHW', 'UNP', 'TXT', 'COST', 'STT',
                         'CLX', 'OXY', 'APA', 'MO', 'MRK', 'AAPL', 'PEG', 'THC', 'MSFT', 'CL', 'FE', 'D', 'DE',
                         'EOG', 'WAT', 'CF', 'HES', 'ADM']"""


dji = pd.read_csv('Stock_data_all_sp500/^DJI_data.csv')
dji['Date'] = pd.to_datetime(dji['Date'])

sp500 = pd.read_csv('Stock_data_all_sp500/^SP500_data.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])

dates = dji['Date']

for key, stocks in stock_dict.items():
    Results = trade.initialize_trading(stocks)

    daily_balances = make_date_column(Results.daily_balances, dates)

    dji = dji.loc[0:len(daily_balances)]
    dji.set_index('Date', inplace=True)

    sp500 = sp500.loc[0:len(daily_balances)]
    sp500.set_index('Date', inplace=True)

    plt.plot(daily_balances)
    "plt.plot(val_0)"
    "plt.plot(val_1)"
    plt.plot(dji['Adj Close'] * 100000 / dji.iloc[0]['Adj Close'])
    plt.plot(sp500['Adj Close'] * 100000 / sp500.iloc[0]['Adj Close'])
    plt.legend(['Predictions', 'Dow Jones Industrial Average', 'S&P 500'])
    plt.title(f'Portfolio value')
    plt.show()

    print(f'Max drawdown for predictions: {get_max_drawdown(Results.daily_balances)}')
    print(f'Max drawdown for sp500: {get_max_drawdown(sp500["Adj Close"])}')
    print(f'Max drawdown for sji: {get_max_drawdown(dji["Adj Close"])}')

    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')

    returns_list_sp500 = np.array(sp500['Adj Close'], dtype=float)
    returns_list_sp500 = np.diff(returns_list_sp500) / returns_list_sp500[:-1]
    print(
        f'Sharpe ratio on sp500: {np.mean(returns_list_sp500) * np.sqrt(252) / np.std(returns_list_sp500)}')
    returns_list_sp500 = list(returns_list_sp500)
    returns_list_sp500_2 = np.array([i if i < 0 else 0 for i in returns_list_sp500])
    print(
        f'Sortino ratio on sp500: {np.mean(returns_list_sp500) * np.sqrt(252) / np.std(returns_list_sp500_2)}')

    returns_list_dji = np.array(dji['Adj Close'], dtype=float)
    returns_list_dji = np.diff(returns_list_dji) / returns_list_dji[:-1]
    print(
        f'Sharpe ratio on dji: {np.mean(returns_list_dji) * np.sqrt(252) / np.std(returns_list_dji)}')
    returns_list_dji = list(returns_list_dji)
    returns_list_dji_2 = np.array([i if i < 0 else 0 for i in returns_list_dji])
    print(
        f'Sortino ratio on dji: {np.mean(returns_list_dji) * np.sqrt(252) / np.std(returns_list_dji_2)}')

predictions = daily_balances.rename(columns={0:'Adj Close'})
predictions = predictions[predictions.index < datetime.datetime(2020, 5, 1)]
