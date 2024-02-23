import datetime
import OptionTradingFuntions as fun
import numpy as np
import pandas as pd
import OptionTradingFuntions as opt
import random
import simplejson


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

        self.blocked = 0

        with open("good_stocks.json", "r") as f1:
            self.good_stocks = simplejson.load(f1)

        with open("stocks_lists_for_each_change.json", "r") as f2:
            self.stocks_lists_for_each_change = simplejson.load(f2)

        self.stocks_data_df = pd.read_csv('sp500_close_data.csv')
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
            data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')

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

                        if self.stocks_owned[symbol] < 0:
                            shares_to_buy_shorts = min(-self.stocks_owned[symbol], shares_to_buy)
                            shares_to_buy -= shares_to_buy_shorts
                            cash_balance -= shares_to_buy_shorts * current_price * (1 + self.transaction_cost)
                            cash_balance += shares_to_buy_shorts * current_price * 1.5
                            self.blocked -= shares_to_buy_shorts * current_price * 1.5
                            self.stocks_owned[symbol] += shares_to_buy_shorts

                        if (self.stocks_owned[symbol] > 0) & (shares_to_buy > 0):
                            while cash_balance < shares_to_buy * current_price * (1 + self.transaction_cost):
                                shares_to_buy -= 1
                            if shares_to_buy < 0:
                                shares_to_buy = 0

                        cash_balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                        self.stocks_owned[symbol] += shares_to_buy

                elif decision == "SELL":
                    invest_amount = cash_to_spend_day * kelly_fraction

                    if invest_amount > 0:
                        shares_to_sell = min(invest_amount // current_price, self.stocks_owned[symbol])

                        if self.stocks_owned[symbol] > 0:
                            shares_to_sell_long = min(self.stocks_owned[symbol], shares_to_sell)
                            shares_to_sell -= shares_to_sell_long
                            cash_balance += shares_to_sell_long * current_price * (1 - self.transaction_cost)
                            self.stocks_owned[symbol] -= shares_to_sell_long

                        if (self.stocks_owned[symbol] <= 0) & (shares_to_sell > 0):
                            while cash_balance < shares_to_sell * current_price * (1 - self.transaction_cost):
                                shares_to_sell -= 1
                            if shares_to_sell < 0:
                                shares_to_sell = 0

                            cash_balance += shares_to_sell * current_price * (1 - self.transaction_cost)
                            cash_balance -= shares_to_sell * current_price * 1.5
                            self.blocked += shares_to_sell * current_price * 1.5
                            self.stocks_owned[symbol] -= shares_to_sell

                stocks_prices[symbol] = current_price
                daily_balance += self.stocks_owned[symbol] * current_price

            daily_balance += cash_balance
            daily_balance += self.blocked

            self.df_decisions, self.stocks_owned, cash_balance, self.blocked = fun.stop_loss(self.stocks_symbols,
                                                                                             self.df_decisions,
                                                                                             self.stocks_data,
                                                                                             day,
                                                                                             self.timedelta,
                                                                                             self.stocks_owned,
                                                                                             cash_balance,
                                                                                             self.transaction_cost,
                                                                                             self.last_prices,
                                                                                             self.blocked)

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

    def update_stocks(self, timedelta):
        """sp_500_historic = pd.read_csv('sp_500_historic_stocks.csv')
        df = pd.DataFrame()
        date_for_stocks_start_train = self.start_date_train + datetime.timedelta(days=1)
        while df.empty:
            date_for_stocks_start_train -= datetime.timedelta(days=1)
            df = sp_500_historic[sp_500_historic['date'] == str(date_for_stocks_start_train)[0:10]]

        list_stocks_start_train_day = [stock for stock in df.values.flatten().tolist()[2:] if str(stock) != 'nan']

        date_next = self.end_date_predict - datetime.timedelta(days=1)
        while df.empty:
            date_next += datetime.timedelta(days=1)
            df = sp_500_historic[sp_500_historic['date'] == str(date_next)[0:10]]

        list_stocks_next = [stock for stock in df.values.flatten().tolist()[2:] if str(stock) != 'nan']

        stocks_in_both = list(set(list_stocks_start_train_day).intersection(list_stocks_next))
        good_stocks_start_train = self.good_stocks[str(date_for_stocks_start_train)[0:10]]
        stocks_in_both = list(set(stocks_in_both).intersection(good_stocks_start_train))

        sp_500_historic_close = pd.read_csv('sp500_close_data.csv')
        stocks_check_nan = sp_500_historic_close[
            (sp_500_historic_close['Date'] >= str(date_for_stocks_start_train)[0:10]) & (
                        sp_500_historic_close['Date'] >= str(date_next)[0:10])][stocks_in_both].isna().sum()

        stocks_good_data = [stock for stock in stocks_check_nan.index.values if stocks_check_nan[stock] == 0]

        stocks_in_both = list(set(stocks_in_both).intersection(stocks_good_data))

        investment_back = {}

        for stock in stocks_in_both:
            data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')
            data = data[(data['Date'] >= str(self.start_date_predict - datetime.timedelta(days=30 * 3))[0:11]) & (
                    data['Date'] <= str(self.start_date_predict)[0:11])]
            data = data.reset_index(drop=True)
            if data.empty:
                investment_back[stock] = 0
            else:
                value_list = np.array(data['Close'])
                value_list = np.diff(value_list) / value_list[:-1]
                investment_back[stock] = np.mean(value_list) * np.sqrt(61) / np.std(value_list)

        investment_back_sorted = {k: v for k, v in sorted(investment_back.items(), key=lambda item: item[1])}
        best_investment = list(investment_back_sorted)[-31: -1]"""
        best_investment = self.stocks_lists_for_each_change[timedelta]

        is_in_current_not_in_new = list(set(self.stocks_symbols) - set(best_investment))
        is_in_new_not_in_current = list(set(best_investment) - set(self.stocks_symbols))
        is_in_both = list(set(best_investment).intersection(self.stocks_symbols))
        new_stock_amounts = {}

        for stock in is_in_new_not_in_current:
            new_stock_amounts[stock] = 0

        for stock in is_in_both:
            new_stock_amounts[stock] = self.stocks_owned[stock]

        money_sold = 0

        for stock in is_in_current_not_in_new:
            amount = self.stocks_owned[stock]
            value = self.stocks_prices_history.iloc[-1][stock]
            money_sold += amount * value

        self.stocks_owned = new_stock_amounts
        self.stocks_symbols = best_investment
        self.daily_cash[-1] += money_sold
        self.df_decisions = pd.DataFrame(columns=self.stocks_symbols)
        self.df_kelly = pd.DataFrame(columns=self.stocks_symbols)
