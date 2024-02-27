import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import trading_initialize as trade
import datetime

matplotlib.use('TkAgg')
yf.pdr_override()

matplotlib.pyplot.yscale('log')

stock_list_numbers = ['stocks 1', 'validation portfolio 1.1', 'validation portfolio 1.2']

stock_dict = {'sharpe': ['PPL', 'NEE', 'BR', 'MKC', 'NTRS', 'KO', 'YUM', 'SCHW', 'UNP', 'TXT', 'COST', 'STT',
                         'CLX', 'OXY', 'APA', 'MO', 'MRK', 'AAPL', 'PEG', 'THC', 'MSFT', 'CL', 'FE', 'D', 'DE',
                         'EOG', 'WAT', 'CF', 'HES', 'ADM'],
              'return': ['CL', 'BK', 'AFL', 'D', 'EP', 'PEG', 'MRK', 'FE', 'MCK', 'YUM', 'X', 'COST', 'UNP',
                         'NTRS', 'TXT', 'SCHW', 'GPS', 'BR', 'WAT', 'STT', 'EOG', 'OXY', 'APA', 'MSFT', 'UNH',
                         'AAPL', 'DE', 'ADM', 'THC', 'HES']
              }
start_day = datetime.datetime(2008, 1, 1)
end_day = datetime.datetime(2021, 1, 1)
dji = pd.DataFrame(yf.download('^DJI', start=start_day, end=end_day)).reset_index()
for key, stocks in stock_dict.items():
    Results, validation_portfolios = trade.initialize_trading(stocks)
    plt.plot(Results.daily_balances)
    """plt.plot(validation_portfolios[0])
    plt.plot(validation_portfolios[1])
    plt.legend(stock_list_numbers)
    plt.title(stock_list)"""
    plt.plot(dji['Close'] * 100000 / 11971)
    plt.legend(['Predictions', 'Dow Jones Industrial Average'])
    plt.title(f'Portfolio value with stocks picked by {key}')
    plt.show()
    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')
