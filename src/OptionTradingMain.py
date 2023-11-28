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

stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'TSLA', 'META', 'SMCI', 'BRK-B', 'LLY', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ', 'AMGN']
stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'META', 'SMCI', 'AMGN', 'LLY', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ']
# stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD']
# stocks_symbols = []

# Run the multi-stock trading simulation
daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash = trade.initialize_trading(stocks_symbols)
plt.plot(daily_balances)
plt.show()

plt.plot(stocks_owned_history)
plt.show()

returns_list = np.array(daily_balances, dtype=float)
returns_list = np.diff(returns_list) / returns_list[:-1]
print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(126) / np.std(returns_list)}')
returns_list = list(returns_list)
returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(126) / np.std(returns_list_2)}')
