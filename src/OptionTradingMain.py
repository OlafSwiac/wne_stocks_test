import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import OptionTradingFuntions as opt
import matplotlib
import matplotlib.pyplot as plt

import TradingInitialize as trade

matplotlib.use('TkAgg')
yf.pdr_override()

matplotlib.pyplot.yscale('log')


# stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'TSLA', 'META', 'SMCI', 'BRK-B', 'LLY', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ', 'AMGN']
# stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'SMCI', 'AMGN', 'LLY', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ']
stocks_symbols = ['AIZ', 'JBHT', 'LIN', 'A', 'AIG', 'SNPS', 'IBM', 'WHR', 'NOC', 'NVR', 'DRI', 'BEN', 'LEN', 'QCOM',
                  'EOG', 'HAS', 'SRE']

stock_list_numbers = ['stocks 1', 'stocks 2', 'stocks 3', 'stocks 4']

stock_list = [['ILMN', 'TTWO', 'CMG', 'HES', 'MS', 'PARA', 'NVR', 'D', 'NDSN', 'ETN', 'TEL', 'WY'],
              ['CHD', 'DLR', 'EXC', 'SO', 'BWA', 'BALL', 'KR', 'MPWR', 'NEE', 'AMP', 'MTB', 'WRB'],
              ['RSG', 'MLM', 'SYY', 'VLO', 'KLAC', 'ON', 'SPGI', 'AMT', 'SNA', 'JCI', 'EIX', 'PAYX'],
              ['BRO', 'CAH', 'BAX', 'PWR', 'SBAC', 'PGR', 'WELL', 'JNJ', 'IRM', 'EA']
]

"""['MRO', 'MTD', 'PFG', 'ADBE', 'WDC', 'MHK', 'WHR', 'CB', 'MCK'],
    ['LNT', 'TJX', 'SLB', 'JNJ', 'PH', 'ROL', 'TER', 'PPL', 'TECH', 'BSX', 'FIS', 'ACGL', 'GEN'],
    ['GWW', 'FMC', 'NUE', 'MSFT', 'NFLX', 'DTE', 'COST', 'ZION', 'BG'],
    ['GWW', 'FMC', 'LNT', 'TJX', 'ADBE', 'WDC', 'MHK', 'SLB', 'JNJ', 'PH', 'ROL', 'TER', 'PPL'],
    ['AIG', 'SNPS', 'IBM', 'WHR', 'NOC', 'NVR', 'DRI', 'BEN', 'LEN', 'QCOM', 'EOG', 'HAS', 'SRE'],
    ['AIZ', 'JBHT', 'LIN', 'A', 'AIG', 'SLB', 'JNJ', 'PH', 'ROL', 'NVR', 'FMC', 'NUE', ],
    ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ']"""

results = []

for stocks in stock_list:
    daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash = trade.initialize_trading(
        stocks)
    results.append([daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash])
    plt.plot(daily_balances)
    plt.show()
# stocks_symbols = []
plt.legend(stock_list_numbers)
# Run the multi-stock trading simulation
"""daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash = trade.initialize_trading(
    stocks_symbols)

plt.plot(daily_balances)
plt.show()"""

"""plt.plot(stocks_owned_history)
plt.show()"""

returns_list = np.array(daily_balances, dtype=float)
returns_list = np.diff(returns_list) / returns_list[:-1]
print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
returns_list = list(returns_list)
returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')
