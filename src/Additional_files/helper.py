import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.metrics import PerformanceMetrics
import datetime

list_of_files = {'default': '../results_csv/default.csv',
            'tc_0005': '../results_csv/tc_005.csv',
            'tc_001': '../results_csv/tc_001.csv',
            'tc_0': '../results_csv/tc_0.csv',
            'sl_l_3': '../results_csv/sl_l_3.csv',
            'sl_l_5': '../results_csv/sl_l_5.csv',
            'no_sl': '../results_csv/no_sl.csv',
            'kelly_f2': '../results_csv/kelly_f2.csv',
            'kelly_f3': '../results_csv/kelly_f3.csv',
            'vg_1.5': '../results_csv/vg_1.5.csv',
            'vg_2.5': '../results_csv/vg_2.5.csv',
            'no_vg': '../results_csv/no_vg.csv',
            'best_IR_2': '../results_csv/best_IR_2.csv',
            'second_ASD': '../results_csv/second_ASD.csv',
            'sp500': '../sp500.csv',
            'dji': '../dji.csv',
            'val_2': '../results_csv/validation_stop_loss.csv',
            'val_1': '../results_csv/validation_no_stop_loss.csv',
            'sl_s_0.5': '../results_csv/sl_s_05.csv',
            'sl_s_1.5': '../results_csv/sl_s_15.csv'
            }
results = {}

for key, value in list_of_files.items():
    results[key] = pd.read_csv(value)
    results[key]['Date'] = pd.to_datetime(results[key]['Date'])
    results[key].set_index('Date', inplace=True)


default_2010 = results['default'][results['default'].index >= datetime.datetime(2010, 1, 1)]
val_1_2010 = results['val_1'][results['val_1'].index >= datetime.datetime(2010, 1, 1)]
val_2_2010 = results['val_2'][results['val_2'].index >= datetime.datetime(2010, 1, 1)]
dji_2010 = results['dji'][results['dji'].index >= datetime.datetime(2010, 1, 1)]
sp500_2010 = results['sp500'][results['sp500'].index >= datetime.datetime(2010, 1, 1)]

"""plt.plot(results['sp500'])
plt.plot(results['dji'], color='forestgreen')
plt.legend(['S&P 500', 'Dow Jones Industrial Average'])"""


