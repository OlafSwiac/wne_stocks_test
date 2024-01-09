import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt

import TradingInitialize as trade

matplotlib.use('TkAgg')
yf.pdr_override()

matplotlib.pyplot.yscale('log')

stock_list_numbers = ['stocks 1', 'validation portfolio 1.1', 'validation portfolio 1.2']

stock_list = [['LHX', 'NUE', 'NEM', 'EA', 'JNPR', 'APA', 'R', 'DHI', 'PDCO']]

for stocks in stock_list:
    Results, validation_portfolios = trade.initialize_trading(stocks)
    plt.plot(Results.daily_balances)
    plt.plot(validation_portfolios[0])
    plt.plot(validation_portfolios[1])
    plt.legend(stock_list_numbers)
    plt.title(stock_list)
    plt.show()
    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')
