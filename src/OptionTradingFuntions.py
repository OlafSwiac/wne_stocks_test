import keras.optimizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Define the function to create lagged features
def create_lagged_features(df_to_change, n_lags=20):
    """Create lagged features for previous n_lags days."""
    df = df_to_change.copy()
    for i in range(1, n_lags + 1):
        df[f'day_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    """df['Close'] = (df['Close'] - df['day_1']) / df['day_1']
    for i in range(1, n_lags):
        df[f'day_{i}'] = (df[f'day_{i}'] - df[f'day_{i + 1}']) / df[f'day_{i + 1}']
    df.drop(columns=[f'day_{n_lags}'], inplace=True)"""
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

    predicted_prices, predicted_test_prices, y_test = get_ann_model(data_train, data_test, data_predict)
    percent_win = max(min(get_percent_wins(data_train, predicted_test_prices, y_test) + np.random.normal(scale=0.2, loc=0), 1), 0)
    for i in range(0, len(predicted_prices)):
        predicted_price = predicted_prices[i][0]
        yesterday_price = data_predict[['Close']].iloc[i + prediction_days].values[0]

        if predicted_price > yesterday_price:
            decisions.append('BUY')
            kelly_fraction = (predicted_price - yesterday_price) * (2 * percent_win - 1) + np.random.normal(scale=0.05, loc=0)
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))
        if predicted_price < yesterday_price:
            decisions.append('SELL')
            kelly_fraction = (yesterday_price - predicted_price) * (2 * percent_win - 1) + np.random.normal(scale=0.05, loc=0)
            kelly_fractions.append(max(min(kelly_fraction, 1), 0))

    return kelly_fractions, decisions


def get_ann_model(x_train: pd.DataFrame, x_test: pd.DataFrame, X_predict: pd.DataFrame):
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

    """scaled_data_train = np.reshape(scaled_data_train, (scaled_data_train.shape[0], scaled_data_train.shape[1], 1))
    scaled_data_test = np.reshape(scaled_data_test, (scaled_data_test.shape[0], scaled_data_test.shape[1], 1))
    scaled_data_predict = np.reshape(scaled_data_predict,
                                     (scaled_data_predict.shape[0], scaled_data_predict.shape[1], 1))"""
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    scaler_prices = StandardScaler()
    y_train = scaler_prices.fit_transform(y_train)

    """regressor = RandomForestRegressor(n_estimators=10192, random_state=1, criterion='absolute_error', min_samples_leaf=10, min_samples_split=0.2, verbose=1, max_features=1, oob_score=True, n_jobs=-1)
    regressor.fit(scaled_data_train, y_train)"""

    svr = SVR(kernel='poly', C=9, gamma='scale', epsilon=0.001, degree=5, coef0=0.85)
    svr.fit(scaled_data_train, y_train)


    """class PlotLosses(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []

            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1

            clear_output(wait=True)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            # plt.show();

    plot_losses = PlotLosses()

    # learning rate control
    initial_learning_rate = 0.8
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=10,
        decay_rate=0.5,
        staircase=True)

    model = Sequential()

    model.add(SimpleRNN(units=256, return_sequences=True, input_shape=(scaled_data_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(SimpleRNN(units=64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(Dense(256, activation='relu'))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(Dense(256, activation='relu'))
    model.add(SimpleRNN(units=15))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=SGD(learning_rate=lr_schedule, clipnorm=1), loss=['mean_absolute_error'])
    model.fit(scaled_data_train, y_train, epochs=256, batch_size=64, verbose=1, callbacks=[plot_losses])
    plt.show()"""
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
