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
from sklearn.ensemble import HistGradientBoostingClassifier

"""matplotlib.use('TkAgg')"""


def create_lagged_features(df_to_change, n_lags=20):
    """Create lagged features for previous n_lags days."""
    df = df_to_change.copy()
    for i in range(1, n_lags + 1):
        df[f'day_{i}'] = df['Adj Close'].shift(i)
    df.dropna(inplace=True)
    return df


def get_percent_wins(data_train: pd.DataFrame, predicted_test_prices: pd.DataFrame, y_test: pd.DataFrame):
    wins = 0
    for i in range(0, len(y_test)):
        if (predicted_test_prices[len(y_test) - 1][0] - data_train['Adj Close'].iloc[-i]) * (
                y_test[len(y_test) - 1][0] - data_train['Adj Close'].iloc[-i]) > 0:
            wins += 1
    return wins / len(y_test)


def get_predictions_and_kelly_criterion(data_train: pd.DataFrame, data_test: pd.DataFrame, data_predict: pd.DataFrame,
                                        prediction_days: int):
    decisions = []
    kelly_fractions = []

    """predicted_prices, predicted_test_prices, y_test = get_prediction_model(data_train, data_test, data_predict,
                                                                           prediction_days)
    percent_win = max(
        min(get_percent_wins(data_train, predicted_test_prices, y_test), 1), 0)
    for i in range(0, len(predicted_prices)):
        predicted_price = predicted_prices[i][0]
        yesterday_price = data_predict[['Adj Close']].iloc[i + prediction_days].values[0]

        if predicted_price > yesterday_price:
            decisions.append('BUY')
            kelly_fraction = ((predicted_price - yesterday_price) * (2 * percent_win - 1)) ** 2
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))
        if predicted_price < yesterday_price:
            decisions.append('SELL')
            kelly_fraction = ((yesterday_price - predicted_price) * (2 * percent_win - 1)) ** 2
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))"""

    kelly_fractions, decisions = classification_test(data_train, data_test, data_predict, prediction_days)

    return kelly_fractions, decisions


def classification_test(data_train: pd.DataFrame, data_test: pd.DataFrame, data_predict: pd.DataFrame,
                        prediction_days: int):
    data_train = create_lagged_features(data_train[['Adj Close']], prediction_days)
    data_test = create_lagged_features(data_test[['Adj Close']], prediction_days)
    data_predict = create_lagged_features(data_predict[['Adj Close']], prediction_days).drop(['Adj Close'], axis=1)

    data_train.rename(columns={'Adj Close': 'Decision'}, inplace=True)
    data_test.rename(columns={'Adj Close': 'Decision'}, inplace=True)

    data_train['Decision'] = data_train.apply(lambda x: 'BUY' if x['Decision'] > x['day_1'] else 'SELL', axis=1)
    data_test['Decision'] = data_test.apply(lambda x: 'BUY' if x['Decision'] > x['day_1'] else 'SELL', axis=1)

    y_train = data_train['Decision']
    y_test = data_test['Decision']

    data_train = data_train.drop(['Decision'], axis=1)
    data_test = data_test.drop(['Decision'], axis=1)

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(np.array(data_train))
    scaled_data_test = scaler.transform(np.array(data_test))
    scaled_data_predict = scaler.transform(np.array(data_predict))

    models = []
    predicted_probabilities = []
    scores = []

    """models.append(RandomForestClassifier(n_estimators=64, criterion='gini', max_features=None, class_weight='balanced'))
    models.append(GaussianProcessClassifier(optimizer='fmin_l_bfgs_b', multi_class='one_vs_rest', max_iter_predict=100, n_restarts_optimizer=0))
    models.append(GradientBoostingClassifier(criterion='squared_error', learning_rate=0.05, loss='exponential', max_features='log2', n_estimators=75))
    models.append(DecisionTreeClassifier(splitter='random', min_weight_fraction_leaf=0.01, min_samples_split=4, criterion='gini', class_weight=None))"""
    models.append(HistGradientBoostingClassifier())

    for model in models:
        model.fit(scaled_data_train, y_train)
        predicted_probabilities.append(model.predict_proba(scaled_data_predict))
        scores.append(model.score(scaled_data_test, y_test))

    kelly = []
    decisions_predicted = []

    for i in range(len(predicted_probabilities[0])):
        proba = 0
        for j in range(len(models)):
            proba += predicted_probabilities[j][i][0]
        proba = proba / len(models)

        if proba > 0.5:
            kelly.append(proba)
            decisions_predicted.append('BUY')
        else:
            kelly.append(1 - proba)
            decisions_predicted.append('SELL')

    return kelly, decisions_predicted


def stop_loss(stocks_symbols, stocks_decisions, stocks_data, day, timedelta, stocks_owned, cash_balance,
              transaction_cost, last_prices, blocked, price_bought):
    for symbol in stocks_symbols:
        current_low = stocks_data[symbol].iloc[day]['Low'] * stocks_data[symbol].iloc[day]['Adj Close'] / \
                      stocks_data[symbol].iloc[day]['Close']
        current = stocks_data[symbol].iloc[day]['Adj Close']
        open_price = stocks_data[symbol].iloc[day]['Open'] * stocks_data[symbol].iloc[day]['Adj Close'] / \
                     stocks_data[symbol].iloc[day]['Close']
        if day > 1:
            previous_price = stocks_data[symbol].iloc[day - 1]['Adj Close']
        elif last_prices != 'DAY ONE':
            previous_price = last_prices[symbol]
        else:
            previous_price = current
        if price_bought[symbol] == 0:
            price_change = 0
        else:
            price_change = (current - price_bought[symbol]) / price_bought[symbol]

        # stop los 0.5%
        if (price_change < -0.015) & (stocks_owned[symbol] > 0):
            print(f'stop_loss: {symbol}, period / day: {timedelta} / {day}, price change {price_change}')
            cash_balance += stocks_owned[symbol] * current * (1 - transaction_cost)
            stocks_owned[symbol] = 0
            price_bought[symbol] = 0
            # print(stocks_decisions.at[day, symbol])
            stocks_decisions.at[day, symbol] = 'SELL'

        """if (price_change > 0.02) & (stocks_owned[symbol] < 0):
            print(f'stop_loss: {symbol}, period / day: {timedelta} / {day}, price change {price_change}')
            cash_balance -= -stocks_owned[symbol] * current * (1 + transaction_cost)
            blocked -= -stocks_owned[symbol] * current * 1.5
            cash_balance += -stocks_owned[symbol] * current * 1.5
            stocks_owned[symbol] = 0
            # print(stocks_decisions.at[day, symbol])
            stocks_decisions.at[day, symbol] = 'BUY'"""
    return stocks_decisions, stocks_owned, cash_balance, blocked


def remake_kelly(stocks_symbols, stocks_kelly_fractions, stocks_decisions):
    for day in stocks_decisions.index:
        sum_buy_kelly = 0
        sum_sell_kelly = 0
        for stock in stocks_symbols:
            if stocks_decisions.at[day, stock] == 'BUY':
                sum_buy_kelly += stocks_kelly_fractions.at[day, stock]
            if stocks_decisions.at[day, stock] == 'SELL':
                sum_sell_kelly += stocks_kelly_fractions.at[day, stock]

        for stock in stocks_symbols:
            if (stocks_decisions.at[day, stock] == 'BUY') & (sum_buy_kelly > 0):
                stocks_kelly_fractions.at[day, stock] = stocks_kelly_fractions.at[day, stock] / sum_buy_kelly
    return stocks_kelly_fractions


def get_validation_portfolio_month(stocks_symbols, start_of_trading, val_1, val_2):
    end_of_trading = start_of_trading + datetime.timedelta(days=30)
    price_data_all = pd.DataFrame()

    for stock_symbol in stocks_symbols:
        data = pd.read_csv(f'Stock_data_all_sp500/{stock_symbol}_data.csv')
        data = data[(data['Date'] >= str(start_of_trading)[0:11]) & (data['Date'] <= str(end_of_trading)[0:11])]
        price_data_all[stock_symbol] = data['Adj Close'].reset_index(drop=True)

    # validation portfolio 1
    sum_price_day_0 = price_data_all.iloc[0].sum()
    amount = val_1 // sum_price_day_0
    validation_portfolio_1 = []
    left_v1 = val_1 - sum_price_day_0 * amount

    for i in range(len(price_data_all)):
        validation_portfolio_1.append(price_data_all.iloc[i].sum() * amount + left_v1)

    # validation portfolio 2
    money_for_each_stocks = val_2 / len(stocks_symbols)
    amount_for_each_stock = {}
    left_v2 = val_2
    for stock_symbol in stocks_symbols:
        amount_for_each_stock[stock_symbol] = money_for_each_stocks // price_data_all.iloc[0][stock_symbol] if \
            price_data_all.iloc[0][stock_symbol] else 0
        left_v2 -= amount_for_each_stock[stock_symbol] * price_data_all.iloc[0][stock_symbol] if price_data_all.iloc[0][
            stock_symbol] else 0

    validation_portfolio_2 = []
    for i in range(len(price_data_all)):
        sum_of_portfolio = 0
        for stock_symbol in stocks_symbols:
            sum_of_portfolio += price_data_all.iloc[i][stock_symbol] * amount_for_each_stock[stock_symbol]

        validation_portfolio_2.append(sum_of_portfolio + left_v2)

    return [validation_portfolio_1, validation_portfolio_2]


def get_validation_portfolios(stocks_start, stock_lists, periods, initial_date):
    stocks = stocks_start
    validations_full = [[100000], [100000]]
    for timedelta in range(periods):
        if (timedelta + 1) % 3 == 0:
            stocks = stock_lists[str(timedelta)]

        start_of_trading = initial_date + datetime.timedelta(days=4 * 365 + 2 * 30 + timedelta * 30)
        validations_month = get_validation_portfolio_month(stocks, start_of_trading, validations_full[0][-1],
                                                           validations_full[1][-1])
        validations_full[0] = validations_full[0] + validations_month[0]
        validations_full[1] = validations_full[1] + validations_month[1]

    return validations_full


def get_max_dropdown(portfolio_values: list):
    MD = 0
    last_max = 0
    for i in range(len(portfolio_values)):
        if portfolio_values[i] > last_max:
            last_max = portfolio_values[i]
        else:
            MD = max(MD, (last_max - portfolio_values[i]) / last_max)
    return MD


def make_date_column(results: list, dates: pd.DataFrame):
    df = pd.DataFrame(results)
    df['Date'] = dates
    df.set_index('Date', inplace=True)
    return df
