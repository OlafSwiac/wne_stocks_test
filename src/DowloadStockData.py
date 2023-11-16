import yfinance as yf
import datetime
import pandas as pd

start_day = datetime.datetime(2012, 1, 1)
end_day = datetime.datetime(2020, 1, 1)
stocks_symbols = ['MSFT', 'NKE', 'INTC', 'AAPL', 'GOOGL', 'AMZN', 'GME', 'AMD', 'TSLA', 'META', 'SMCI']

for stock in stocks_symbols:
    data = pd.DataFrame(yf.download(stock, start=start_day, end=end_day))
    data.to_csv(f'{stock}_data.csv')
