import numpy as np
import pandas as pd
import datetime

sp_500_historic = pd.read_csv('sp_500_historic_stocks.csv')

start_day = datetime.datetime(2004, 1, 1)
three_month_before = start_day - datetime.timedelta(days=30 * 3)
four_years_before = start_day - datetime.timedelta(days=365 * 4)

stocks = []
df = pd.DataFrame()

date_for_stocks = start_day
while df.empty:
    df = sp_500_historic[sp_500_historic['date'] == str(date_for_stocks)[0:10]]
    date_for_stocks -= datetime.timedelta(days=1)

stocks = df.values.flatten().tolist()
date = stocks[1]
stocks = stocks[2:]
stocks_now = [stock for stock in stocks if str(stock) != 'nan']

stocks = []
df = pd.DataFrame()

date_for_stocks = four_years_before
while df.empty:
    df = sp_500_historic[sp_500_historic['date'] == str(date_for_stocks)[0:10]]
    date_for_stocks -= datetime.timedelta(days=1)

stocks = df.values.flatten().tolist()
date = stocks[1]
stocks = stocks[2:]
stocks_four_yb = [stock for stock in stocks if str(stock) != 'nan']

stocks_in_both = list(set(stocks_now).intersection(stocks_four_yb))

investment_back = {}

for stock in stocks_in_both:
    data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')
    data = data[(data['Date'] >= str(three_month_before)[0:11]) & (
            data['Date'] <= str(start_day)[0:11])]
    data = data.reset_index(drop=True)
    if data.empty:
        investment_back[stock] = 0
    else:
        """v1 = data.iloc[0]['Close']
        v2 = data.iloc[-1]['Close']
        investment_back[stock] = (v2 - v1) / v1"""
        value_list = np.array(data['Close'])
        value_list = np.diff(value_list) / value_list[:-1]
        investment_back[stock] = np.mean(value_list) * np.sqrt(61) / np.std(value_list)

investment_back_sorted = {k: v for k, v in sorted(investment_back.items(), key=lambda item: item[1])}
best_investment = list(investment_back_sorted)[-31: -1]
