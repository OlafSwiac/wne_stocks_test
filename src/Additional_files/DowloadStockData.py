import yfinance as yf
import datetime
import pandas as pd
from Scrap_sp500 import get_random_stocks
import simplejson

start_day = datetime.datetime(2005, 2, 1)
end_day = datetime.datetime(2023, 10, 1)
"""data = pd.DataFrame(yf.download('USDT-USD', start=start_day, end=end_day))
data.to_csv(f'../Stock_data_all_sp500/USDT-USD_data.csv')"""



"""with open('../Stock_lists/good_stocks.json', "r") as f3:
    good = simplejson.load(f3)

all_stocks = set()

for key, value in good.items():
    all_stocks.update(value)

for stock in all_stocks:
    data = pd.DataFrame(yf.download(stock, start=start_day, end=end_day))
    data.to_csv(f'../Stock_data_all_sp500/{stock}_data.csv')
"""

data = pd.DataFrame(yf.download('^GSPC', start=start_day, end=end_day))
data.to_csv(f'../Stock_data_all_sp500/^SP500_data.csv')

data_2 = pd.DataFrame(yf.download('^DJI', start=start_day, end=end_day))
data_2.to_csv(f'../Stock_data_all_sp500/^DJI_data.csv')