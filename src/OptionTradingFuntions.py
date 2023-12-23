import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
import datetime
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso

matplotlib.use('TkAgg')


def create_lagged_features(df_to_change, n_lags=20):
    """Create lagged features for previous n_lags days."""
    df = df_to_change.copy()
    for i in range(1, n_lags + 1):
        df[f'day_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df


def get_percent_wins(data_train: pd.DataFrame, predicted_test_prices: pd.DataFrame, y_test: pd.DataFrame):
    wins = 0
    for i in range(0, len(y_test)):
        if (predicted_test_prices[len(y_test) - 1][0] - data_train['Close'].iloc[-i]) * (
                y_test[len(y_test) - 1][0] - data_train['Close'].iloc[-i]):
            wins += 1
    # print(wins / len(y_test))
    return wins / len(y_test)


def get_predictions_and_kelly_criterion(data_train: pd.DataFrame, data_test: pd.DataFrame, data_predict: pd.DataFrame,
                                        prediction_days: int):
    decisions = []
    kelly_fractions = []

    predicted_prices, predicted_test_prices, y_test = get_prediction_model(data_train, data_test, data_predict,
                                                                           prediction_days)
    percent_win = max(
        min(get_percent_wins(data_train, predicted_test_prices, y_test) + np.random.normal(scale=0.2, loc=0), 1), 0)
    for i in range(0, len(predicted_prices)):
        predicted_price = predicted_prices[i][0]
        yesterday_price = data_predict[['Close']].iloc[i + prediction_days].values[0]

        if predicted_price > yesterday_price:
            decisions.append('BUY')
            kelly_fraction = (predicted_price - yesterday_price) * (2 * percent_win - 1) + np.random.normal(scale=0.05,
                                                                                                            loc=0)
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))
        if predicted_price < yesterday_price:
            decisions.append('SELL')
            kelly_fraction = (yesterday_price - predicted_price) * (2 * percent_win - 1) + np.random.normal(scale=0.05,
                                                                                                            loc=0)
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))

    return kelly_fractions, decisions


def get_prediction_model(x_train: pd.DataFrame, x_test: pd.DataFrame, X_predict: pd.DataFrame, prediction_days):
    data_train = create_lagged_features(x_train[['Close']], prediction_days)
    data_test = create_lagged_features(x_test[['Close']], prediction_days)
    data_predict = create_lagged_features(X_predict[['Close']], prediction_days).drop(['Close'], axis=1)

    y_train = data_train['Close']
    y_test = data_test['Close']
    data_train = data_train.drop(['Close'], axis=1)
    data_test = data_test.drop(['Close'], axis=1)

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(np.array(data_train))
    scaled_data_test = scaler.transform(np.array(data_test))
    scaled_data_predict = scaler.transform(np.array(data_predict))

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    scaler_prices = StandardScaler()
    y_train = scaler_prices.fit_transform(y_train)

    svr = SVR(kernel='poly', C=8, gamma='scale', epsilon=0.01, degree=3, coef0=0.95)
    svr.fit(scaled_data_train, y_train)

    br = BayesianRidge()
    br.fit(scaled_data_train, y_train)

    lasso = Lasso(alpha=0.004)
    lasso.fit(scaled_data_train, y_train)

    # predicted_prices = model.predict(scaled_data_predict)
    predicted_prices_svr = scaler_prices.inverse_transform(svr.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_svr = scaler_prices.inverse_transform(svr.predict(scaled_data_test).reshape(-1, 1))

    predicted_prices_br = scaler_prices.inverse_transform(br.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_br = scaler_prices.inverse_transform(br.predict(scaled_data_test).reshape(-1, 1))

    predicted_prices_lasso = scaler_prices.inverse_transform(lasso.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_lasso = scaler_prices.inverse_transform(lasso.predict(scaled_data_test).reshape(-1, 1))

    predicted_prices = (predicted_prices_br * 1 / 3 + predicted_prices_svr * 1 / 3 + predicted_prices_lasso * 1 / 3)
    predicted_test_prices = (
            predicted_test_prices_br * 1 / 3 + predicted_test_prices_svr * 1 / 3 + predicted_test_prices_lasso * 1 / 3)

    predicted_prices_perc = []

    for i in range(1, len(predicted_prices)):
        predicted_prices_perc.append((predicted_prices[i][0] - predicted_prices[i - 1][0]) / predicted_prices[i - 1][0])

    return predicted_prices, predicted_test_prices, y_test


def stop_loss(stocks_symbols, stocks_decisions, stocks_data, day, timedelta, stocks_owned, cash_balance,
              transaction_cost, last_prices):
    for symbol in stocks_symbols:
        current_price = stocks_data[symbol].iloc[day]['Close']
        open_price = stocks_data[symbol].iloc[day]['Open']
        if day > 1:
            previous_price = stocks_data[symbol].iloc[day - 1]['Close']
        elif last_prices != 'DAY ONE':
            previous_price = last_prices[symbol]
        else:
            previous_price = current_price

        price_change = (current_price - previous_price) / previous_price

        # stop los 2%
        if (price_change < -0.02) & (stocks_owned[symbol] > 0):
            print(f'stop_loss: {symbol}, period / day: {timedelta} / {day}, price change {price_change}')
            cash_balance += stocks_owned[symbol] * min(previous_price, open_price) * (1 - 0.02) * (1 - transaction_cost)
            stocks_owned[symbol] = 0
            # print(stocks_decisions.at[day, symbol])
            stocks_decisions.at[day, symbol] = 'SELL'
    return stocks_decisions, stocks_owned, cash_balance


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


def get_validation_portfolio(stocks_symbols, initial_time, periods):
    start_of_trading = initial_time + datetime.timedelta(days=4 * 365 + 2 * 30)
    end_of_trading = initial_time + datetime.timedelta(days=4 * 365 + 2 * 30 + (periods - 1) * 30)
    price_data_all = pd.DataFrame()

    for stock_symbol in stocks_symbols:
        data = pd.read_csv(f'Stock_Data/{stock_symbol}_data.csv')
        data = data[(data['Date'] >= str(start_of_trading)[0:11]) & (data['Date'] <= str(end_of_trading)[0:11])]
        price_data_all[stock_symbol] = data['Close'].reset_index(drop=True)

    # validation portfolio 1
    sum_price_day_0 = price_data_all.iloc[0].sum()
    amount = 100000 // sum_price_day_0
    validation_portfolio_1 = []

    for i in range(len(price_data_all)):
        validation_portfolio_1.append(price_data_all.iloc[i].sum() * amount)

    # validation portfolio 2
    amount_of_stocks = 100000 / len(stocks_symbols)
    amount_for_each_stock = {}
    for stock_symbol in stocks_symbols:
        amount_for_each_stock[stock_symbol] = amount_of_stocks // price_data_all.iloc[0][stock_symbol]

    validation_portfolio_2 = []
    for i in range(len(price_data_all)):
        sum_of_portfolio = 0
        for stock_symbol in stocks_symbols:
            sum_of_portfolio += price_data_all.iloc[i][stock_symbol] * amount_for_each_stock[stock_symbol]

        validation_portfolio_2.append(sum_of_portfolio)

    return [validation_portfolio_1, validation_portfolio_2]
