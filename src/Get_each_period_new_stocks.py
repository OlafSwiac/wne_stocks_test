import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import simplejson

sp_500_historic = pd.read_csv('src/sp_500_historic_stocks.csv')
sp_500_historic = sp_500_historic.fillna('')
sp_500_all = set()
good = {}

list_of_good_stocks = list(pd.read_csv('src/sp500_close_data.csv').columns[1:])

for row in sp_500_historic.index:
    list_sp500_row = sp_500_historic.iloc[row].to_list()[1:]
    list_sp500_row = [i for i in list_sp500_row if i != '']
    date_index = list_sp500_row[0]
    good_list = []
    for stock in list_sp500_row:
        if stock in list_of_good_stocks:
            good_list.append(stock)
    good[date_index] = good_list

with open("good_stocks.json", "w") as fp:
    simplejson.dump(good, fp)

print(0)
""""start_day = datetime.datetime(2000, 1, 1)


df = pd.DataFrame()
while df.empty:
    df = sp_500_historic[sp_500_historic['date'] == str(start_day)[0:10]]
    start_day -= datetime.timedelta(days=1)

list_stocks_start_day = df.iloc[0].to_list()[1:]
list_stocks_start_day = [i for i in list_stocks_start_day if i != '']

random_symbols = set()

for i in range(0, 20):
    random_symbols.update([list_stocks_start_day[np.random.random_integers(0, len(list_stocks_start_day))]])

random_symbols_list = list(random_symbols)
"""
