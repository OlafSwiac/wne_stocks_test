import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import simplejson

start_date_train = datetime.datetime(2004, 1, 1)
end_date_train = start_date_train + datetime.timedelta(days=4 * 365)
start_date_test = start_date_train + datetime.timedelta(days=4 * 365 - 30)
end_date_test = end_date_train + datetime.timedelta(days=20)
start_date_predict = end_date_train
end_date_predict = start_date_predict + datetime.timedelta(days=2 * 30)

with open("../Stock_lists/good_stocks.json", "r") as f1:
    good_stocks = simplejson.load(f1)

sp_500_historic = pd.read_csv('../sp_500_historic_stocks.csv')
df = pd.DataFrame()
date_for_stocks_start_train = start_date_train + datetime.timedelta(days=1)
while df.empty:
    date_for_stocks_start_train -= datetime.timedelta(days=1)
    df = sp_500_historic[sp_500_historic['date'] == str(date_for_stocks_start_train)[0:10]]

list_stocks_start_train_day = [stock for stock in df.values.flatten().tolist()[2:] if str(stock) != 'nan']

date_next = end_date_predict - datetime.timedelta(days=1)
while df.empty:
    date_next += datetime.timedelta(days=1)
    df = sp_500_historic[sp_500_historic['date'] == str(date_next)[0:10]]

list_stocks_next = [stock for stock in df.values.flatten().tolist()[2:] if str(stock) != 'nan']

stocks_in_both = list(set(list_stocks_start_train_day).intersection(list_stocks_next))
good_stocks_start_train = good_stocks[str(date_for_stocks_start_train)[0:10]]
stocks_in_both = list(set(stocks_in_both).intersection(good_stocks_start_train))

sp_500_historic_close = pd.read_csv('../sp500_close_data.csv')
stocks_check_nan = sp_500_historic_close[
    (sp_500_historic_close['Date'] >= str(date_for_stocks_start_train)[0:10]) & (
                sp_500_historic_close['Date'] >= str(date_next)[0:10])][stocks_in_both].isna().sum()

stocks_good_data = [stock for stock in stocks_check_nan.index.values if stocks_check_nan[stock] == 0]

stocks_in_both = list(set(stocks_in_both).intersection(stocks_good_data))

investment_back = {}

for stock in stocks_in_both:
    data = pd.read_csv(f'../Stock_data_all_sp500/{stock}_data.csv')
    data = data[(data['Date'] >= str(start_date_predict - datetime.timedelta(days=30 * 3))[0:11]) & (
            data['Date'] <= str(start_date_predict)[0:11])]
    data = data.reset_index(drop=True)
    if data.empty:
        investment_back[stock] = 0
    else:
        """value_list = np.array(data['Close'])
        value_list = np.diff(value_list) / value_list[:-1]
        investment_back[stock] = np.mean(value_list) * np.sqrt(61) / np.std(value_list)"""
        investment_back[stock] = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]

investment_back_sorted = {k: v for k, v in sorted(investment_back.items(), key=lambda item: item[1])}
best_investment = list(investment_back_sorted)[-31: -1]
print(best_investment)