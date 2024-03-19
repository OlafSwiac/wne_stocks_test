import datetime
import simplejson

initial_time=datetime.datetime(2004, 1, 1)

with open("../Stock_lists/stocks_lists_20_for_each_change_sharpe.json", "r") as f2:
    stocks_lists_for_each_change = simplejson.load(f2)

stocks = ['PPL', 'NEE', 'BR', 'MKC', 'NTRS', 'KO', 'YUM', 'SCHW', 'UNP', 'TXT', 'COST', 'STT',
                         'CLX', 'OXY', 'APA', 'MO', 'MRK', 'AAPL', 'PEG', 'THC']

for timedelta in range(144):
    start_date_train = initial_time + datetime.timedelta(days=30 * timedelta)
    end_date_train = start_date_train + datetime.timedelta(days=4 * 365)
    start_date_test = start_date_train + datetime.timedelta(days=4 * 365 - 30)
    end_date_test = end_date_train + datetime.timedelta(days=20)
    start_date_predict = end_date_train
    end_date_predict = start_date_predict + datetime.timedelta(days=2 * 30)

    if (timedelta + 1) % 3:
        stocks = stocks_lists_for_each_change[timedelta]

    
