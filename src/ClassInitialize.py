import datetime
import OptionTradingFuntions as fun
import numpy as np
import pandas as pd
import OptionTradingFuntions as opt
import random


class TradingAlgorithmEnvironment:
    def __init__(self, stocks_symbols=[], initial_time=datetime.datetime(2004, 1, 1), daily_cash=[100000],
                 stocks_owned={}, prediction_days=20, transaction_cost=0.0005, daily_balances=[], final_balance=0,
                 stocks_owned_history=pd.DataFrame(), stocks_prices_history=pd.DataFrame()):
        self.stocks_symbols = stocks_symbols
        self.initial_time = initial_time
        self.timedelta = 0
        self.start_date_train = self.initial_time + datetime.timedelta(days=30 * self.timedelta)
        self.end_date_train = self.start_date_train + datetime.timedelta(days=4 * 365)
        self.start_date_test = self.start_date_train + datetime.timedelta(days=4 * 365 - 30)
        self.end_date_test = self.end_date_train + datetime.timedelta(days=20)
        self.start_date_predict = self.end_date_train
        self.end_date_predict = self.start_date_predict + datetime.timedelta(days=2 * 30)

        self.daily_balances = daily_balances
        self.final_balance = final_balance
        self.daily_cash = daily_cash
        self.stocks_data = {}
        self.df_decisions = pd.DataFrame(columns=stocks_symbols)
        self.df_kelly = pd.DataFrame(columns=stocks_symbols)

        self.prediction_days = prediction_days
        self.transaction_cost = transaction_cost
        self.stocks_owned_history = stocks_owned_history
        self.stocks_prices_history = stocks_prices_history
        self.stocks_owned = stocks_owned

        self.last_prices = np.NAN

    def update_data(self):
        self.start_date_train = self.initial_time + datetime.timedelta(days=30 * self.timedelta)
        self.end_date_train = self.start_date_train + datetime.timedelta(days=4 * 365)
        self.start_date_test = self.start_date_train + datetime.timedelta(days=4 * 365 - 30)
        self.end_date_test = self.end_date_train + datetime.timedelta(days=20)
        self.start_date_predict = self.end_date_train
        self.end_date_predict = self.start_date_predict + datetime.timedelta(days=2 * 30)

        self.df_decisions = pd.DataFrame(columns=self.stocks_symbols)
        self.df_kelly = pd.DataFrame(columns=self.stocks_symbols)
        self.stocks_data = {}

        for stock in self.stocks_symbols:
            data = pd.read_csv(f'Stock_Data/{stock}_data.csv')

            data_train = data[(data['Date'] >= str(self.start_date_train)[0:11]) & (
                    data['Date'] <= str(self.end_date_train)[0:11])]

            data_test = data[(data['Date'] >= str(self.start_date_test)[0:11]) & (
                    data['Date'] <= str(self.end_date_test)[0:11])]

            data_predict = data[(data['Date'] >= str(self.start_date_predict)[0:11]) & (
                    data['Date'] <= str(self.end_date_predict)[0:11])]

            self.df_kelly[stock], self.df_decisions[stock] = \
                opt.get_predictions_and_kelly_criterion(data_train, data_test, data_predict, self.prediction_days)
            self.stocks_data[stock] = data_train
            del data_train
            del data_test
            del data_predict

    def update_timedelta(self, timedelta=0):
        self.timedelta = timedelta

    def update_last_prices(self, last_prices=0):
        self.last_prices = last_prices

    def simulate_multi_stock_trading(self):
        self.df_kelly = fun.remake_kelly(self.stocks_symbols, self.df_kelly, self.df_decisions)
        stocks_prices = {symbol: 0 for symbol in self.stocks_symbols}
        daily_cash = []
        cash_balance = self.daily_cash[-1]
        random.shuffle(self.stocks_symbols)
        cash_to_spend_day = cash_balance


        for day in self.df_decisions.index:
            daily_balance = 0

            # Iterate through each symbol for the current day
            for symbol in self.stocks_symbols:
                decision = self.df_decisions.at[day, symbol]
                kelly_fraction = self.df_kelly.at[day, symbol]
                current_price = self.stocks_data[symbol].iloc[day]['Close']

                if (decision == "BUY") & (cash_balance > 0):
                    invest_amount = cash_to_spend_day * kelly_fraction

                    invest_amount *= 1 - self.transaction_cost

                    if invest_amount > 0:
                        shares_to_buy = max(min(invest_amount // current_price, 0.3 * cash_balance // current_price),
                                            0)

                        while cash_balance < shares_to_buy * current_price * (1 + self.transaction_cost):
                            shares_to_buy -= 1
                        if shares_to_buy < 0:
                            shares_to_buy = 0

                        cash_balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                        self.stocks_owned[symbol] += shares_to_buy

                elif (decision == "SELL") & (self.stocks_owned[symbol] > 0):
                    invest_amount = cash_to_spend_day * kelly_fraction

                    if invest_amount > 0:
                        shares_to_sell = min(invest_amount // current_price, self.stocks_owned[symbol])
                        cash_balance += shares_to_sell * current_price * (1 - self.transaction_cost)
                        self.stocks_owned[symbol] -= shares_to_sell

                stocks_prices[symbol] = current_price
                daily_balance += self.stocks_owned[symbol] * current_price

            daily_balance += cash_balance

            self.df_decisions, self.stocks_owned, cash_balance = fun.stop_loss(self.stocks_symbols,
                                                                               self.df_decisions, self.stocks_data,
                                                                               day,
                                                                               self.timedelta, self.stocks_owned,
                                                                               cash_balance,
                                                                               self.transaction_cost, self.last_prices)

            self.daily_balances.append(daily_balance)
            daily_cash.append(cash_balance)
            self.stocks_owned_history = pd.concat([self.stocks_owned_history, pd.DataFrame([self.stocks_owned])],
                                                  ignore_index=True)
            self.stocks_prices_history = pd.concat([self.stocks_prices_history, pd.DataFrame([stocks_prices])],
                                                   ignore_index=True)
            print(f'daily balance - period / day: {self.timedelta} / {day}, {daily_balance} \n')
            if self.timedelta == 22:
                print("ERROR")
            self.daily_cash.append(cash_balance)

        self.final_balance = self.daily_balances[-1]

    def update_stocks(self):
        sp_500_historic = pd.read_csv('sp_500_historic_stocks.csv')
        df = pd.DataFrame()
        date_for_stocks = self.start_date_train
        while df.empty:
            df = sp_500_historic[sp_500_historic['date'] == str(date_for_stocks)[0:10]]
            date_for_stocks -= datetime.timedelta(days=1)

        list_stocks_start_day = df.iloc[0].to_list()[1:]
        list_stocks_start_day = [i for i in list_stocks_start_day if i != '']

        random_symbols = set()

        for i in range(0, 20):
            random_symbols.update([list_stocks_start_day[np.random.random_integers(0, len(list_stocks_start_day))]])

        self.stocks_symbols = list(random_symbols)
