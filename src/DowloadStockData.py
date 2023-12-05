import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt

start_day = datetime.datetime(2000, 1, 1)
end_day = datetime.datetime(2022, 1, 1)
stocks_symbols = ['KR', 'PGR', 'WRB', 'EIX', 'TTWO', 'SBAC', 'SPGI', 'RSG', 'CMG', 'GM', 'PAYX', 'AMP', 'JCI', 'IQV', 'ILMN', 'EXC', 'BALL', 'JNJ', 'HCA', 'EA', 'ETN', 'CMG', 'SO', 'NEE', 'ADI', 'CFG', 'CAH', 'BWA', 'HES', 'KLAC', 'SNA', 'NDSN', 'CHD', 'TEL', 'LYB', 'PARA', 'MS', 'MTB', 'AMT', 'BRO', 'PWR', 'ON', 'IQV', 'WELL', 'WY', 'D', 'MLM', 'SNA', 'BAX', 'HSIC', 'LYB', 'FTNT', 'HPE', 'MPWR', 'DLR', 'VLO', 'HES', 'SYY', 'NVR', 'IRM']
for stock in stocks_symbols:
    data = pd.DataFrame(yf.download(stock, start=start_day, end=end_day))
    data.to_csv(f'Stock_Data/{stock}_data.csv')

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
