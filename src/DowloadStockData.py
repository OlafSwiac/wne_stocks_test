import yfinance as yf
import datetime
import pandas as pd

start_day = datetime.datetime(2000, 1, 1)
end_day = datetime.datetime(2022, 1, 1)
stocks_symbols = ['^RUT']

for stock in stocks_symbols:
    data = pd.DataFrame(yf.download(stock, start=start_day, end=end_day))
    data.to_csv(f'Stock_Data/{stock}_data.csv')

"""import requests

url = "https://api.marketdata.app/v1/options/quotes/AAPL250117C00150000/?date=2023-01-18"

response = requests.request("GET", url)

print(response.text)"""
