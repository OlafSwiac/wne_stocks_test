import pandas as pd
import matplotlib.pyplot as plt
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
            'sp500': '../results_csv/sp500.csv',
            'dji': '../results_csv/dji.csv',
            'val_2': '../results_csv/validation_stop_loss.csv',
            'val_1': '../results_csv/validation_no_stop_loss.csv',
            'sl_s_0.5': '../results_csv/sl_s_05.csv',
            'sl_s_1.5': '../results_csv/sl_s_15.csv',
            'each_month': '../results_csv/each_month.csv',
            'each_4_months': '../results_csv/each_4_months.csv',
            'each_3_months': '../results_csv/each_3_months.csv'
            }
results = {}

for key, value in list_of_files.items():
    results[key] = pd.read_csv(value)
    results[key]['Date'] = pd.to_datetime(results[key]['Date'])
    results[key].set_index('Date', inplace=True)


results['default_2010'] = results['default'][results['default'].index >= datetime.datetime(2010, 1, 1)]
results['val_1_2010'] = results['val_1'][results['val_1'].index >= datetime.datetime(2010, 1, 1)]
results['val_2_2010'] = results['val_2'][results['val_2'].index >= datetime.datetime(2010, 1, 1)]
results['dji_2010'] = results['dji'][results['dji'].index >= datetime.datetime(2010, 1, 1)]
results['sp500_2010'] = results['sp500'][results['sp500'].index >= datetime.datetime(2010, 1, 1)]

"""plt.plot(results['sp500'])
plt.plot(results['dji'], color='forestgreen')
plt.legend(['S&P 500', 'Dow Jones Industrial Average'])"""

def make_table(results: dict, list_of_columns: list, table_file: str):
    table = {}
    for column in list_of_columns:
        table[column] = PerformanceMetrics(results[column])
    table_df = pd.DataFrame(table)
    table_df.to_csv(table_file, sep='&', lineterminator=' \\\ [1ex]\n')

make_table(results, ['default', 'no_vg'], '../tables/test.csv')
make_table(results, ['default', 'val_1', 'val_2', 'sp500', 'dji'], '../tables/main_table.csv')
make_table(results, ['default_2010', 'val_1_2010', 'val_2_2010', 'sp500_2010', 'dji_2010'], '../tables/from_2010_table.csv')
make_table(results, ['default', 'tc_0005', 'tc_001', 'tc_0'], '../tables/sens_tc.csv')
make_table(results, ['default', 'sl_l_3', 'sl_l_5', 'no_sl'], '../tables/sens_sl_long.csv')
make_table(results, ['default', 'sl_s_0.5', 'sl_s_1.5'], '../tables/sens_sl_short.csv')
make_table(results, ['default', 'kelly_f2', 'kelly_f3'], '../tables/sens_kelly.csv')
make_table(results, ['default', 'vg_1.5', 'vg_2.5', 'no_vg'], '../tables/sens_vg.csv')
make_table(results, ['default', 'best_IR_2', 'second_ASD'], '../tables/sens_metric.csv')
make_table(results, ['default', 'each_month', 'each_3_months', 'each_4_months'], '../tables/sens_period.csv')




