import numpy as np
# import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import trading_initialize as trade
import datetime
from trading_functions import get_max_drawdown, make_date_column
import metrics as m
import seaborn as sns

"""matplotlib.use('TkAgg')
# yf.pdr_override()

matplotlib.pyplot.yscale('log')"""

stock_dict = {
    'second_IR_2': ["NWL", "IBM", "BAX", "ADP", "MDT", "MMM", "WFC", "PNW", "WM", "PFG", "FE", "ETR", "VFC", "PBI",
                    "PG"],
    'IR_2': ["AAPL", "AIV", "AMD", "ATI", "CSX", "FLR", "GLW", "GT", "IGT", "LEG", "SLB", "SRE", "TJX", "UNP", "VIAV"],
    'ASD': ["AFL", "CAG", "CPB", "FE", "GE", "GPC", "JPM", "L", "MMM", "MSFT", "PFG", "TROW", "WFC", "WY", "ZION"],
    'btc': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'USDT-USD']}

strategy = {'default': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                        'file': 'results_csv/default.csv', 'stocks': stock_dict['second_IR_2']},
            'tc_0005': {'tc': 0.0005, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                        'file': 'results_csv/tc_005.csv', 'stocks': stock_dict['second_IR_2']},
            'tc_001': {'tc': 0.001, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                       'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                       'file': 'results_csv/tc_001.csv', 'stocks': stock_dict['second_IR_2']},
            'tc_0': {'tc': 0, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                     'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                     'file': 'results_csv/tc_0.csv', 'stocks': stock_dict['second_IR_2']},
            'sl_l_3': {'tc': 0.00075, 'sl_long': 0.03, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                       'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                       'file': 'results_csv/sl_l_3.csv', 'stocks': stock_dict['second_IR_2']},
            'sl_l_5': {'tc': 0.00075, 'sl_long': 0.05, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                       'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                       'file': 'results_csv/sl_l_5.csv', 'stocks': stock_dict['second_IR_2']},
            'no_sl': {'tc': 0.00075, 'sl_long': 10000, 'sl_short': 10000, 'kelly': 'f1', 'vg': 2,
                      'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                      'file': 'results_csv/no_sl.csv', 'stocks': stock_dict['second_IR_2']},
            'kelly_f2': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f2', 'vg': 2,
                         'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                         'file': 'results_csv/kelly_f2.csv', 'stocks': stock_dict['second_IR_2']},
            'kelly_f3': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f3', 'vg': 2,
                         'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                         'file': 'results_csv/kelly_f3.csv', 'stocks': stock_dict['second_IR_2']},
            'vg_1.5': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 1.5,
                       'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                       'file': 'results_csv/vg_1.5.csv', 'stocks': stock_dict['second_IR_2']},
            'vg_2.5': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2.5,
                       'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                       'file': 'results_csv/vg_2.5.csv', 'stocks': stock_dict['second_IR_2']},
            'no_vg': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 10000,
                      'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                      'file': 'results_csv/no_vg.csv', 'stocks': stock_dict['second_IR_2']},
            'best_IR_2': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                          'metrics': 'Stock_lists/new_stocks_test_15_best_IR_2_2005_train.json',
                          'file': 'results_csv/best_IR_2.csv', 'stocks': stock_dict['IR_2']},
            'second_ASD': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                           'metrics': 'Stock_lists/new_stocks_test_15_second_best_ASD_2005_train.json',
                           'file': 'results_csv/second_ASD.csv', 'stocks': stock_dict['ASD']},
            'sl_s_05': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.005, 'kelly': 'f1', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                        'file': 'results_csv/sl_s_05.csv', 'stocks': stock_dict['second_IR_2']},
            'sl_s_15': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.015, 'kelly': 'f1', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                        'file': 'results_csv/sl_s_15.csv', 'stocks': stock_dict['second_IR_2']}
            }

strategy = {'crypto': {'tc': 0.00075, 'sl_long': 0.02, 'sl_short': 0.005, 'kelly': 'f2', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json',
                        'file': 'results_csv/default_exp.csv', 'stocks': stock_dict['btc']}}

strategy = {'default': {'tc': 0.00075, 'sl_long': 0.04, 'sl_short': 0.01, 'kelly': 'f1', 'vg': 2,
                        'metrics': 'Stock_lists/new_stocks_test_15_second_best_IR_2_2005_train.json'}}

dji = pd.read_csv('Stock_data_all_sp500/^DJI_data.csv')
dji['Date'] = pd.to_datetime(dji['Date'])

sp500 = pd.read_csv('Stock_data_all_sp500/^SP500_data.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])

dates = dji['Date']
Results = {}
predictions = {}
daily_balances = {}
for key, parameters in strategy.items():
    Results[key] = trade.initialize_trading(parameters)

    #daily_balances[key] = make_date_column(Results[key].daily_balances, dates)

    """dji = dji.loc[0:len(daily_balances)]
    dji.set_index('Date', inplace=True)

    sp500 = sp500.loc[0:len(daily_balances)]
    sp500.set_index('Date', inplace=True)"""

    """plt.plot(daily_balances)
    "plt.plot(val_0)"
    "plt.plot(val_1)"
    plt.plot(dji['Adj Close'] * 100000 / dji.iloc[0]['Adj Close'])
    plt.plot(sp500['Adj Close'] * 100000 / sp500.iloc[0]['Adj Close'])
    plt.legend(['Predictions', 'Dow Jones Industrial Average', 'S&P 500'])
    plt.title(f'Portfolio value')
    plt.show()

    print(f'Max drawdown for predictions: {get_max_drawdown(Results.daily_balances)}')
    print(f'Max drawdown for sp500: {get_max_drawdown(sp500["Adj Close"])}')
    print(f'Max drawdown for sji: {get_max_drawdown(dji["Adj Close"])}')

    returns_list = np.array(Results.daily_balances, dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    print(f'Sharpe ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list)}')
    returns_list = list(returns_list)
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    print(f'Sortino ratio: {np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)}')

    returns_list_sp500 = np.array(sp500['Adj Close'], dtype=float)
    returns_list_sp500 = np.diff(returns_list_sp500) / returns_list_sp500[:-1]
    print(
        f'Sharpe ratio on sp500: {np.mean(returns_list_sp500) * np.sqrt(252) / np.std(returns_list_sp500)}')
    returns_list_sp500 = list(returns_list_sp500)
    returns_list_sp500_2 = np.array([i if i < 0 else 0 for i in returns_list_sp500])
    print(
        f'Sortino ratio on sp500: {np.mean(returns_list_sp500) * np.sqrt(252) / np.std(returns_list_sp500_2)}')

    returns_list_dji = np.array(dji['Adj Close'], dtype=float)
    returns_list_dji = np.diff(returns_list_dji) / returns_list_dji[:-1]
    print(
        f'Sharpe ratio on dji: {np.mean(returns_list_dji) * np.sqrt(252) / np.std(returns_list_dji)}')
    returns_list_dji = list(returns_list_dji)
    returns_list_dji_2 = np.array([i if i < 0 else 0 for i in returns_list_dji])
    print(
        f'Sortino ratio on dji: {np.mean(returns_list_dji) * np.sqrt(252) / np.std(returns_list_dji_2)}')"""

    #predictions[key] = daily_balances[key].rename(columns={0: 'Adj Close'})
    #predictions[key] = predictions[key][predictions[key].index < datetime.datetime(2020, 1, 1)]
    """val_1 = pd.DataFrame(Results[key].validation_portfolio)
    val_2 = pd.DataFrame(Results[key].validation_portfolio_stop_loss)
    val_1 = val_1.rename(columns={0:'Adj Close'})
    val_2 = val_2.rename(columns={0:'Adj Close'})
    val_1 = make_date_column(val_1, dates)
    val_2 = make_date_column(val_2, dates)
    val_1 = val_1[val_1.index < datetime.datetime(2020, 1, 1)]
    val_2 = val_2[val_2.index < datetime.datetime(2020, 1, 1)]"""

    """sp500 = sp500[['Adj Close']]
    dji = dji[['Adj Close']]
    sp500 = sp500[sp500.index < datetime.datetime(2020, 1, 1)]
    sp500 = sp500 * 100000 / sp500.loc[datetime.datetime(2006, 2, 1)]['Adj Close']
    dji = dji[dji.index < datetime.datetime(2020, 1, 1)]
    dji = dji * 100000 / dji.loc[datetime.datetime(2006, 2, 1)]['Adj Close']"""

    #predictions[key].to_csv(parameters['file'])

"""btc = pd.read_csv('Stock_data_all_sp500/BTC-USD_data.csv')
eth = pd.read_csv('Stock_data_all_sp500/ETH-USD_data.csv')
bnb = pd.read_csv('Stock_data_all_sp500/BNB-USD_data.csv')
usdt = pd.read_csv('Stock_data_all_sp500/USDT-USD_data.csv')


btc['Date'] = pd.to_datetime(btc['Date'])
btc = btc[btc['Date'] >= datetime.datetime(2019, 1, 1)]
eth['Date'] = pd.to_datetime(eth['Date'])
bnb['Date'] = pd.to_datetime(bnb['Date'])
usdt['Date'] = pd.to_datetime(usdt['Date'])
dates = btc['Date']

predictions['crypto'] = make_date_column(Results['crypto'].daily_balances, dates)
predictions['crypto'] = predictions['crypto'].rename(columns={0: 'Adj Close'})

btc.set_index('Date', inplace=True)
eth.set_index('Date', inplace=True)
bnb.set_index('Date', inplace=True)
usdt.set_index('Date', inplace=True)

btc = btc[(btc.index >= datetime.datetime(2019, 1, 1)) & (btc.index < datetime.datetime(2022, 1, 1))]
eth = eth[(eth.index >= datetime.datetime(2019, 1, 1)) & (eth.index < datetime.datetime(2022, 1, 1))]
bnb = bnb[(bnb.index >= datetime.datetime(2019, 1, 1)) & (bnb.index < datetime.datetime(2022, 1, 1))]
usdt = usdt[(usdt.index >= datetime.datetime(2019, 1, 1)) & (usdt.index < datetime.datetime(2022, 1, 1))]

btc = btc[['Adj Close']]
eth = eth[['Adj Close']]
bnb = bnb[['Adj Close']]
usdt = usdt[['Adj Close']]

btc = btc/btc.iloc[0]['Adj Close'] * 100000
eth = eth/eth.iloc[0]['Adj Close'] * 100000
bnb = bnb/bnb.iloc[0]['Adj Close'] * 100000
usdt = usdt/usdt.iloc[0]['Adj Close'] * 100000

sum = (btc + eth + bnb + usdt) / 4

sns.lineplot(predictions['crypto'][0])
sns.lineplot(sum.reset_index()['Adj Close'])"""


