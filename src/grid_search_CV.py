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
from sklearn.metrics import class_likelihood_ratios
from sklearn.ensemble import AdaBoostClassifier
pd.options.mode.chained_assignment = None

def scoring(estimator, X, y):
    y_proba = estimator.predict_proba(X)
    y_pred = []
    for i in range(len(y_proba)):
        if y_proba[i][0] > 0.35:
            y_pred.append('BUY')
        else:
            y_pred.append('SELL')
    pos_lr, neg_lr = class_likelihood_ratios(y, y_pred, raise_warning=False)
    return {"positive_likelihood_ratio": pos_lr, "negative_likelihood_ratio": neg_lr}


prediction_days = 20
stocks = ['COP', 'NUE', 'GD', 'TWX', 'FCX', 'CVS', 'AYE', 'TAP', 'SBUX', 'MO', 'CFC', 'WBA', 'MMM',
          'MSI', 'PG', 'WFC', 'PPL', 'T', 'ETR', 'BSC', 'WEN', 'CCU', 'GPC', 'OMC', 'CSCO', 'PH', 'ODP',
          'BAX', 'OXY', 'FDX', 'CBE', 'MAR', 'MAS', 'CMCSA', 'BC', 'SLR', 'STT', 'SVU', 'KR', 'SLB', 'CMI', 'IBM',
          'ESRX', 'HES', 'CNP', 'ATI', 'AAPL', 'SYY', 'FLR', 'BDX', 'AET', 'EIX', 'XEL', 'JWN', 'AMD', 'NOC', 'HIG',
          'L', 'VFC', 'GT', 'CMS', 'NTRS', 'MCO', 'R', 'ED', 'RF', 'IFF', 'GLW', 'VZ', 'DRI', 'DDS', 'NKE', 'OMX', 'BA',
          'INTC', 'AON', 'PFE', 'BDK', 'ZBH', 'CI', 'THC', 'JBL', 'ITT', 'BBY', 'FE', 'PEG', 'PBG', 'WAT', 'LMT', 'NVDA',
          'MCD', 'MTG', 'AES', 'ADSK', 'JPM', 'PGR', 'ITW', 'GPS', 'RIG', 'MRK', 'XOM', 'CLX', 'CIEN', 'UVN',
          'GR', 'HUM', 'NEE', 'MRO', 'ADM', 'APD', 'CL']

start_train_day = datetime.datetime(2004, 1, 1)
start_test_day = datetime.datetime(2006, 1, 1)
end_test_day = datetime.datetime(2006, 1, 1)

best_params = []
for stock in stocks:
    data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')
    data_train = data[(data['Date'] >= str(start_train_day)[0:11]) & (
            data['Date'] <= str(start_test_day)[0:11])]
    """data_test = data[(data['Date'] >= str(start_test_day)[0:11]) & (
            data['Date'] <= str(end_test_day)[0:11])]"""
    data_train.reset_index(drop=True, inplace=True)
    data_train.reset_index(inplace=True)
    data_train['Percentage'] = data_train.apply(
        lambda x: 100 * (x['Adj Close'] - data_train.loc[max(0, x['index'] - 1)]['Adj Close']) /
                  data_train.loc[max(0, x['index'] - 1)]['Adj Close'], axis=1)

    data_train = create_lagged_features(data_train, 'Percentage', prediction_days)
    """data_test = create_lagged_features(data_test[['Adj Close']], prediction_days)"""

    data_train.rename(columns={'Percentage': 'Decision'}, inplace=True)
    """data_test.rename(columns={'Adj Close': 'Decision'}, inplace=True)"""

    data_train['Decision'] = data_train.apply(lambda x: 'BUY' if x['Decision'] > 0 else 'SELL', axis=1)
    """data_test['Decision'] = data_test.apply(lambda x: 'BUY' if x['Decision'] > x['day_1'] else 'SELL', axis=1)"""

    y_train = data_train['Decision']
    """y_test = data_test['Decision']"""

    data_train = data_train.drop(['Decision'], axis=1)
    """data_test = data_test.drop(['Decision'], axis=1)"""

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(np.array(data_train))
    """scaled_data_test = scaler.transform(np.array(data_test))"""

    adb = AdaBoostClassifier(algorithm='SAMME')
    rfc = RandomForestClassifier(class_weight='balanced')
    gpc = GaussianProcessClassifier()
    hgbc = HistGradientBoostingClassifier()



    parameters_random_forest = {'n_estimators': (128, 256), 'criterion': ('gini', 'entropy', 'log_loss'), 'max_features': ('sqrt', 'log2', 0.5)}
    parameters_gaussian_processes = {'optimizer': ('fmin_l_bfgs_b', None), 'n_restarts_optimizer': (0, 1, 2), 'multi_class': ('one_vs_rest', 'one_vs_one')}
    parameters_ada_boost = {'learning_rate': (0.05, 0.1, 0.15), 'n_estimators': (128, 256)}
    parameters_hist_gradient_boosting = {'learning_rate': (0.05, 0.1, 0.15), 'max_bins': (50, 100, 200), 'interaction_cst': ('pairwise', 'no_interactions'), 'scoring': ('loss', 'accuracy', 'neg_log_loss', 'average_precision'), 'class_weight': ('balanced', None)}

    grid_search = GridSearchCV(rfc, param_grid=parameters_random_forest, scoring='accuracy', cv=5)
    grid_search.fit(scaled_data_train, y_train)

    print(grid_search.best_params_, grid_search.best_score_)
    best_params.append(grid_search.best_params_)

df = pd.DataFrame(best_params)

for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())
