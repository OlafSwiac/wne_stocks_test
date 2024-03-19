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


prediction_days = 20
stocks = ['WMT', 'IGT', 'BK', 'TRB', 'YUM', 'OXY', 'LMT', 'XEL', 'GS', 'BDK', 'AON', 'HUM', 'SCHW', 'D', 'SRE', 'NKE', 'ESRX', 'CFC', 'MCD', 'EOG', 'MO', 'PEP', 'THC', 'KO', 'NTRS', 'MRK', 'PG', 'JNJ', 'WAT', 'CL']
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

    "model = RandomForestClassifier(n_estimators=64, criterion='gini', max_features=None, class_weight='balanced')"
    "model = GaussianProcessClassifier(optimizer='fmin_l_bfgs_b', multi_class='one_vs_rest', max_iter_predict=100, n_restarts_optimizer=0)"
    
    model = DecisionTreeClassifier(splitter='random', min_weight_fraction_leaf=0.01, min_samples_split=4, criterion='gini', class_weight=None)
    "model = GradientBoostingClassifier(criterion='squared_error', learning_rate=0.05, loss='exponential', max_features='log2', n_estimators=75)"

    parameters = {'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy', 'log_loss'), 'min_samples_split': (2, 3, 4), 'min_weight_fraction_leaf': (0, 0.01, 0.02), 'class_weight': ('balanced', None)}

    grid_search = GridSearchCV(model, parameters)
    grid_search.fit(scaled_data_train, y_train)

    print(grid_search.best_params_)
    best_params.append(grid_search.best_params_)

df = pd.DataFrame(best_params)

for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())
