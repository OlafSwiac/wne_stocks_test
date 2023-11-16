import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Define the function to create lagged features
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
    return wins / len(y_test)


def get_predictions_and_kelly_criterion(data_train: pd.DataFrame, data_test: pd.DataFrame, data_predict: pd.DataFrame,
                                        prediction_days: int):
    decisions = []
    kelly_fractions = []

    predicted_prices, predicted_test_prices, y_test = get_prediction_model(data_train, data_test, data_predict)
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


def get_prediction_model(x_train: pd.DataFrame, x_test: pd.DataFrame, X_predict: pd.DataFrame):
    prediction_days = 15

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

    svr = SVR(kernel='poly', C=7, gamma='scale', epsilon=0.001, degree=4, coef0=0.95)

    svr.fit(scaled_data_train, y_train)

    # predicted_prices = model.predict(scaled_data_predict)
    predicted_prices = scaler_prices.inverse_transform(svr.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices = scaler_prices.inverse_transform(svr.predict(scaled_data_test).reshape(-1, 1))

    """predicted_prices_perc = []

    for i in range(1, len(predicted_prices)):
        predicted_prices_perc.append((predicted_prices[i][0] - predicted_prices[i - 1][0]) / predicted_prices[i - 1][0])

    actual_prices_perc = []

    for i in range(15, len(X_predict[['Close']].values)):
        actual_prices_perc.append(
            (X_predict[['Close']].values[i][0] - X_predict[['Close']].values[i - 1][0]) /
            X_predict[['Close']].values[i - 1][0])

    plt.plot(predicted_prices_perc, color='green')
    plt.plot(actual_prices_perc, color='red')
    plt.show()

    plt.plot(predicted_prices, color='green')
    plt.plot(X_predict[['Close']].values[prediction_days:], color='red')
    plt.show()
    plt.clf()"""
    return predicted_prices, predicted_test_prices, y_test


def restoring_balanced_portfolio(stocks_symbols, stocks_owned, stocks_rebalanced, cash_balance, current_price):
    for stock in stocks_symbols:
        stock_difference = stocks_owned[stock] - stocks_rebalanced[stock] - 1
        if stock_difference > 0:
            cash_balance += stock_difference * current_price[stock] * (1 - 0.005)
            stocks_owned[stock] = stocks_rebalanced[stock]
        else:
            cash_balance += stock_difference * current_price[stock] * (1 + 0.005)
            stocks_owned[stock] = stocks_rebalanced[stock]
    return stocks_owned, cash_balance, cash_balance


def counter_the_stabilized_portfolio(stocks_symbols, stocks_owned, cash_balance, decision, kelly_fraction,
                                     current_price):
    portfolio_value = cash_balance
    stocks_rebalanced = {stock: 0 for stock in stocks_symbols}
    for stock in stocks_symbols:
        portfolio_value += stocks_owned[stock] * current_price[stock]

    for stock in stocks_symbols:
        stocks_rebalanced[stock] = (portfolio_value * kelly_fraction[stock]) // \
                                   current_price[stock] if decision[stock] == 'BUY' else 0
    stocks_owned, cash_balance, cash_balance = restoring_balanced_portfolio(stocks_symbols, stocks_owned,
                                                                            stocks_rebalanced, cash_balance,
                                                                            current_price)
    return stocks_owned, cash_balance, cash_balance
