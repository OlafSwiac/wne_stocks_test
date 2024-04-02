def get_prediction_model(x_train: pd.DataFrame, x_test: pd.DataFrame, X_predict: pd.DataFrame, prediction_days):
    data_train = create_lagged_features(x_train[['Adj Close']], prediction_days)
    data_test = create_lagged_features(x_test[['Adj Close']], prediction_days)
    data_predict = create_lagged_features(X_predict[['Adj Close']], prediction_days).drop(['Adj Close'], axis=1)

    data_train_for_perc = data_train

    y_train = data_train['Adj Close']
    y_test = data_test['Adj Close']
    data_train = data_train.drop(['Adj Close'], axis=1)
    data_test = data_test.drop(['Adj Close'], axis=1)

    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(np.array(data_train))
    scaled_data_test = scaler.transform(np.array(data_test))
    scaled_data_predict = scaler.transform(np.array(data_predict))

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    scaler_prices = StandardScaler()
    y_train = scaler_prices.fit_transform(y_train)

    """svr = SVR(kernel='poly', C=8, gamma='scale', epsilon=0.02, degree=5, coef0=0.9)
    svr.fit(scaled_data_train, y_train)"""

    br = BayesianRidge(tol=0.01, alpha_1=0.4, alpha_2=0.4, lambda_1=0.4, lambda_2=0.4)
    br.fit(scaled_data_train, y_train)

    lasso = Lasso(alpha=0.005)
    lasso.fit(scaled_data_train, y_train)

    sgd_1 = SGDRegressor(loss='huber', penalty='elasticnet', alpha=1, learning_rate='invscaling', epsilon=0.05)
    sgd_1.fit(scaled_data_train, y_train)

    sgd_2 = SGDRegressor(loss='squared_error', penalty='elasticnet', alpha=0.8, learning_rate='invscaling',
                         epsilon=0.01)
    sgd_2.fit(scaled_data_train, y_train)

    kernel = DotProduct() + WhiteKernel()

    gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)
    gpr.fit(scaled_data_train, y_train)

    # predicted_prices = model.predict(scaled_data_predict)
    """predicted_prices_svr = scaler_prices.inverse_transform(svr.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_svr = scaler_prices.inverse_transform(svr.predict(scaled_data_test).reshape(-1, 1))
    perc_win_svr = get_percent_wins(data_train_for_perc, predicted_test_prices_svr, y_test)"""

    predicted_prices_br = scaler_prices.inverse_transform(br.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_br = scaler_prices.inverse_transform(br.predict(scaled_data_test).reshape(-1, 1))
    perc_win_br = get_percent_wins(data_train_for_perc, predicted_test_prices_br, y_test)

    predicted_prices_lasso = scaler_prices.inverse_transform(lasso.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_lasso = scaler_prices.inverse_transform(lasso.predict(scaled_data_test).reshape(-1, 1))
    perc_win_lasso = get_percent_wins(data_train_for_perc, predicted_test_prices_lasso, y_test)

    predicted_prices_sgd_1 = scaler_prices.inverse_transform(sgd_1.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_sgd_1 = scaler_prices.inverse_transform(sgd_1.predict(scaled_data_test).reshape(-1, 1))
    perc_win_sgd_1 = get_percent_wins(data_train_for_perc, predicted_test_prices_sgd_1, y_test)

    predicted_prices_sgd_2 = scaler_prices.inverse_transform(sgd_2.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_sgd_2 = scaler_prices.inverse_transform(sgd_2.predict(scaled_data_test).reshape(-1, 1))
    perc_win_sgd_2 = get_percent_wins(data_train_for_perc, predicted_test_prices_sgd_1, y_test)

    predicted_prices_gpr = scaler_prices.inverse_transform(gpr.predict(scaled_data_predict).reshape(-1, 1))
    predicted_test_prices_gpr = scaler_prices.inverse_transform(gpr.predict(scaled_data_test).reshape(-1, 1))
    perc_win_gpr = get_percent_wins(data_train_for_perc, predicted_test_prices_gpr, y_test)

    print(perc_win_gpr)

    """model = Sequential()

    model.add(LSTM(units=200, return_state=True, return_sequences=True, input_shape=(scaled_data_train.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(Dense(units=256))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd_1', loss='mean_squared_error')
    model.fit(scaled_data_train, y_train, epochs=20, batch_size=8, verbose=1)
    predcted_price_lstm = scaler.inverse_transform((model.predict(scaled_data_predict)))"""

    # print(f'Perc wins br: {perc_win_br}, lasso: {perc_win_lasso}')
    """if perc_win_br == 0 and perc_win_lasso == 0:
        print('0 0')"""

    predicted_prices_svr = 0
    perc_win_svr = 0
    predicted_test_prices_svr = 0
    predicted_prices_br = 0
    perc_win_br = 0
    predicted_test_prices_br = 0
    predicted_prices_lasso = 0
    perc_win_lasso = 0
    predicted_test_prices_lasso = 0

    predicted_prices = (
                               predicted_prices_sgd_1 * perc_win_sgd_1 + predicted_prices_sgd_2 * perc_win_sgd_2 + predicted_prices_br * perc_win_br +
                               predicted_prices_lasso * perc_win_lasso + predicted_prices_gpr * perc_win_gpr) / (
                               perc_win_sgd_1 + perc_win_sgd_2 + perc_win_br + perc_win_lasso + 0.001)

    predicted_test_prices = (
                                    predicted_test_prices_sgd_1 * perc_win_sgd_1 + predicted_test_prices_sgd_2 * perc_win_sgd_2 + predicted_test_prices_br * perc_win_br +
                                    predicted_test_prices_lasso * perc_win_lasso + predicted_test_prices_gpr * perc_win_gpr) \
                            / (perc_win_sgd_1 + perc_win_sgd_2 + perc_win_br + perc_win_lasso + perc_win_gpr + 0.001)

    """predicted_prices = predicted_prices_gpr
    predicted_test_prices = predicted_test_prices_gpr"""

    return predicted_prices, predicted_test_prices, y_test