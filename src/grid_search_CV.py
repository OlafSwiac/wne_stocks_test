from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
import datetime
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


"""import tensorflow as tf
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM"""
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from trading_functions import create_lagged_features
from sklearn.ensemble import HistGradientBoostingClassifier


prediction_days = 20
stocks = ['COP', 'NUE', 'GD', 'TWX', 'FCX', 'CVS', 'AYE', 'TAP', 'SBUX', 'MO', 'CFC', 'WBA', 'MMM',
          'MSI', 'PG', 'WFC', 'PPL', 'T', 'ETR', 'BSC', 'WEN', 'CCU', 'GPC', 'OMC', 'CSCO', 'PH', 'ODP',
          'BAX', 'OXY', 'FDX', 'CBE', 'MAR', 'MAS', 'CMCSA', 'BC', 'SLR', 'STT', 'SVU', 'KR', 'SLB', 'CMI', 'IBM',
          'ESRX', 'HES', 'CNP', 'ATI', 'AAPL', 'SYY', 'FLR', 'BDX', 'AET', 'EIX', 'XEL', 'JWN', 'AMD', 'NOC', 'HIG',
          'L', 'VFC', 'GT', 'CMS', 'NTRS', 'MCO', 'R', 'ED', 'RF', 'IFF', 'GLW', 'VZ', 'DRI', 'DDS', 'NKE', 'OMX', 'BA',
          'INTC', 'AON', 'PFE', 'BDK', 'ZBH', 'CI', 'THC', 'JBL', 'ITT', 'BBY', 'FE', 'PEG', 'PBG', 'WAT', 'LMT', 'NVDA',
          'MCD', 'MTG', 'AES', 'ADSK', 'JPM', 'PGR', 'ITW', 'GPS', 'RIG', 'MRK', 'CPWR', 'XOM', 'CLX', 'CIEN', 'UVN',
          'GR', 'HUM', 'NEE', 'MRO', 'ADM', 'APD', 'CL']

start_train_day = datetime.datetime(2004, 1, 1)
start_test_day = datetime.datetime(2008, 1, 1)
end_test_day = datetime.datetime(2008, 1, 1)

best_params = []
for stock in stocks:
    data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')
    data_train = data[(data['Date'] >= str(start_train_day)[0:11]) & (
            data['Date'] <= str(start_test_day)[0:11])]
    """data_test = data[(data['Date'] >= str(start_test_day)[0:11]) & (
            data['Date'] <= str(end_test_day)[0:11])]"""

    data_train = create_lagged_features(data_train[['Adj Close']], prediction_days)
    """data_test = create_lagged_features(data_test[['Adj Close']], prediction_days)"""

    data_train.rename(columns={'Adj Close': 'Decision'}, inplace=True)
    """data_test.rename(columns={'Adj Close': 'Decision'}, inplace=True)"""

    data_train['Decision'] = data_train.apply(lambda x: 'BUY' if x['Decision'] > x['day_1'] else 'SELL', axis=1)
    """data_test['Decision'] = data_test.apply(lambda x: 'BUY' if x['Decision'] > x['day_1'] else 'SELL', axis=1)"""

    y_train = data_train['Decision']
    """y_test = data_test['Decision']"""

    data_train = data_train.drop(['Decision'], axis=1)
    """data_test = data_test.drop(['Decision'], axis=1)"""

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(np.array(data_train))
    """scaled_data_test = scaler.transform(np.array(data_test))"""

    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    gpc = GaussianProcessClassifier()
    gbc = GradientBoostingClassifier()
    hgbc = HistGradientBoostingClassifier()

    parameters_decision_tree = {'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy', 'log_loss'), 'min_samples_split': (2, 3, 4), 'min_weight_fraction_leaf': (0, 0.01, 0.02), 'class_weight': ('balanced', None)}
    parameters_random_forest = {'n_estimators': (64, 128, 256), 'criterion': ('gini', 'entropy', 'log_loss'), 'min_impurity_decrease': (0, 0.1, 0.2), 'max_features': ('sqrt', 'log2', None), 'class_weight': ('balanced', 'balanced_subsample')}
    parameters_gaussian_processes = {'optimizer': ('fmin_l_bfgs_b', None), 'n_restarts_optimizer': (0, 1, 2), 'multi_class': ('one_vs_rest', 'one_vs_one')}
    parameters_gradient_boosting = {'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy', 'log_loss'), 'min_samples_split': (2, 3, 4), 'min_weight_fraction_leaf': (0, 0.01, 0.02), 'class_weight': ('balanced', None)}
    parameters_hist_gradient_boosting = {'learning_rate': (0.05, 0.1, 0.15), 'max_bins': (128, 256, 512), 'interaction_cst': ('pairwise', 'no_interactions'), 'scoring': ('loss', 'accuracy', 'neg_log_loss', 'average_precision'), 'class_weight': ('balanced', None)}

    grid_search = GridSearchCV(gpc, parameters_gaussian_processes, scoring='neg_log_loss')
    grid_search.fit(scaled_data_train, y_train)

    print(grid_search.best_params_, grid_search.best_score_)
    best_params.append(grid_search.best_params_)

df = pd.DataFrame(best_params)

for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())
