import yfinance as yf
import datetime
import pandas as pd
from Scrap_sp500 import get_random_stocks

start_day = datetime.datetime(2008, 1, 1)
end_day = datetime.datetime(2021, 1, 1)
data = pd.DataFrame(yf.download('^DJI', start=start_day, end=end_day))
data.to_csv(f'../Stock_data_all_sp500/^DJI_data.csv')
"""stocks_symbols = [
    'JNPR', 'APA'
]
for stock in stocks_symbols:
    data = pd.DataFrame(yf.download(stock, start=start_day, end=end_day))
    data.to_csv(f'Stock_Data/{stock}_data.csv')
    print(f'Stock: {stock}, {data.iloc[0]}')"""
"""i = 0
fig, ax = plt.subplots(4, 5)
for stock in stocks_symbols:
    stock_plot = pd.read_csv(f'Stock_Data/{stock}_data.csv')
    ax[i // 5, i % 5].plot(stock_plot['Close'])
    i += 1

plt.show()"""

"""import requests

url = "https://api.marketdata.app/v1/options/quotes/AAPL250117C00150000/?date=2023-01-18"

response = requests.request("GET", url)

print(response.text)"""

data = pd.DataFrame(yf.download('^DJI', start=start_day, end=end_day))
