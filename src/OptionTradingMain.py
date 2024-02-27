import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import TradingInitialize as trade
import datetime

matplotlib.use('TkAgg')
yf.pdr_override()

matplotlib.pyplot.yscale('log')

stock_list_numbers = ['stocks 1', 'validation portfolio 1.1', 'validation portfolio 1.2']

stock_list = [['RSG', 'MLM', 'SYY', 'VLO', 'KLAC', 'ON', 'SPGI', 'AMT', 'SNA', 'JCI', 'EIX', 'PAYX']]
start_day = datetime.datetime(2008, 1, 1)
end_day = datetime.datetime(2021, 1, 1)
dji = pd.DataFrame(yf.download('^DJI', start=start_day, end=end_day)).reset_index()
for stocks in stock_list:
    Results, validation_portfolios = trade.initialize_trading(stocks)
    plt.plot(Results.daily_balances)
    """plt.plot(validation_portfolios[0])
    plt.plot(validation_portfolios[1])
    plt.legend(stock_list_numbers)
    plt.title(stock_list)"""
    plt.plot(dji['Close'] * 100000 / 11971)
    plt.legend(['Predictions', 'Dow Jones Industrial Average'])
    plt.show()
    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')
