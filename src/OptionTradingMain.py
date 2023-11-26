import pandas as pd
import yfinance as yf
import datetime
import OptionTradingFuntions as opt
import matplotlib
import matplotlib.pyplot as plt
import TradingInitialize as trade

matplotlib.use('TkAgg')
yf.pdr_override()

# stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'TSLA', 'META', 'SMCI', 'BRK-B', 'LLY', 'TSM', 'UNH', 'WMT', 'MA', 'JNJ', 'AMGN']
stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD']
# stocks_symbols = ['^RUT']

# Run the multi-stock trading simulation
daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash = trade.initialize_trading(stocks_symbols)
plt.plot(daily_balances)
plt.show()

plt.plot(stocks_owned_history)
plt.show()
