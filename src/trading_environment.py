import datetime
import trading_functions as fun
import numpy as np
import pandas as pd
import trading_functions as opt
import random
import simplejson
from trading_functions import get_max_drawdown


class TradingAlgorithmEnvironment:
    def __init__(self, stocks_symbols=[], initial_time=datetime.datetime(2004, 1, 1),
                 daily_cash=[100000], stocks_owned={}, prediction_days=20,
                 transaction_cost=0.00075, daily_balances=[], final_balance=0, price_bought={},
                 stocks_owned_history=pd.DataFrame(), stocks_prices_history=pd.DataFrame(), stocks_file='', short_stocks_file=''):
        self.stocks_symbols = stocks_symbols
        self.initial_time = initial_time
        self.timedelta = 0
        self.start_date_train = self.initial_time + datetime.timedelta(days=30 * self.timedelta)
        self.end_date_train = self.start_date_train + datetime.timedelta(days=365)
        self.start_date_test = self.start_date_train + datetime.timedelta(days=365 - 30)
        self.end_date_test = self.end_date_train + datetime.timedelta(days=20)
        self.start_date_predict = self.end_date_train
        self.end_date_predict = self.start_date_predict + datetime.timedelta(days=30 * 2)
        self.stocks_file = stocks_file

        self.price_bought = price_bought

        for stock in self.stocks_symbols:
            self.price_bought[stock] = 0

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
        self.stocks_data_train = {}
        self.stocks_data_test = {}
        self.blocked = 0
        self.volatility = {}
        self.validation_portfolio = []

        for stock in self.stocks_symbols:
            self.volatility[stock] = 0

        self.stop_loss_count_long_now = 0
        self.stop_loss_count_short_now = 0
        self.stop_loss_count_long_before = 0
        self.stop_loss_count_short_before = 0

        self.is_short_stock = {}
        for stock in self.stocks_symbols:
            self.is_short_stock[stock] = 0

        with open("Stock_lists/good_stocks.json", "r") as f1:
            self.good_stocks = simplejson.load(f1)

        with open(stocks_file, "r") as f2:
            self.stocks_lists_for_each_change = simplejson.load(f2)

        with open(short_stocks_file, "r") as f3:
            self.short_stocks_list_for_each_change = simplejson.load(f3)

        self.stocks_data_df = pd.read_csv('sp500_close_data.csv')
        self.last_prices = np.NAN

    def update_data(self):
        self.start_date_train = self.initial_time + datetime.timedelta(days=30 * self.timedelta)
        self.end_date_train = self.start_date_train + datetime.timedelta(days=365)
        self.start_date_test = self.start_date_train + datetime.timedelta(days=365 - 30)
        self.end_date_test = self.end_date_train + datetime.timedelta(days=20)
        self.start_date_predict = self.end_date_train
        self.end_date_predict = self.start_date_predict + datetime.timedelta(days=2 * 30)

        self.stop_loss_count_long_before = self.stop_loss_count_long_now
        self.stop_loss_count_short_before = self.stop_loss_count_short_now
        self.stop_loss_count_long_now = 0
        self.stop_loss_count_short_now = 0

        "print(f'Stop loss before long {self.stop_loss_count_long_before}, short {self.stop_loss_count_short_before}')"

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

            data_train.reset_index(drop=True, inplace=True)
            data_train.reset_index(inplace=True)
            data_train['Percentage'] = data_train.apply(
                lambda x: 100 * (x['Adj Close'] - data_train.loc[max(0, x['index'] - 1)]['Adj Close']) /
                          data_train.loc[max(0, x['index'] - 1)]['Adj Close'], axis=1)

            data_test.reset_index(drop=True, inplace=True)
            data_test.reset_index(inplace=True)
            data_test['Percentage'] = data_test.apply(
                lambda x: 100 * (x['Adj Close'] - data_test.loc[max(0, x['index'] - 1)]['Adj Close']) /
                          data_test.loc[max(0, x['index'] - 1)]['Adj Close'], axis=1)

            data_predict.reset_index(drop=True, inplace=True)
            data_predict.reset_index(inplace=True)
            data_predict['Percentage'] = data_predict.apply(
                lambda x: 100 * (x['Adj Close'] - data_predict.loc[max(0, x['index'] - 1)]['Adj Close']) /
                          data_predict.loc[max(0, x['index'] - 1)]['Adj Close'], axis=1)

            "print(stock)"
            self.df_kelly[stock], self.df_decisions[stock] = \
                opt.get_predictions_and_kelly_criterion(data_train, data_test, data_predict, self.prediction_days, self.stop_loss_count_long_before, self.stop_loss_count_short_before, self.is_short_stock[stock])
            self.stocks_data[stock] = data_predict
            self.stocks_data_train[stock] = data_train
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
        stocks_prices_month = pd.DataFrame()
        "print(self.is_short_stock)"
        for day in range(len(self.df_decisions)):
            daily_balance = 0

            for stock in self.stocks_symbols:
                stocks_prices[stock] = self.stocks_data[stock].iloc[day + 20]['Adj Close']
            cash_balance = self.block(stocks_prices, cash_balance)

            # Iterate through each stock for the current day
            for stock in self.stocks_symbols:
                decision = self.df_decisions.at[day, stock]
                kelly_fraction = self.df_kelly.at[day, stock] if self.df_kelly.at[day, stock] else 0
                current_price = self.stocks_data[stock].iloc[day + 20]['Adj Close']
                volatility = self.volatility[stock]

                if volatility > 2:
                    if self.stocks_owned[stock] > 0:
                        decision = 'SELL'
                    elif self.stocks_owned[stock] < 0:
                        decision = 'BUY'
                    else:
                        decision = 'HOLD'

                if decision == "BUY":
                    invest_amount = cash_to_spend_day * kelly_fraction
                    if cash_to_spend_day < 1000:
                        invest_amount = self.final_balance * kelly_fraction

                    invest_amount *= 1 - self.transaction_cost

                    if invest_amount > 0:
                        shares_to_buy = max(invest_amount // current_price, 0)
                        if self.is_short_stock[stock]:
                            shares_to_buy = min(shares_to_buy, -self.stocks_owned[stock])

                        if self.stocks_owned[stock] < 0:
                            shares_to_buy_shorts = min(-self.stocks_owned[stock], shares_to_buy)
                            shares_to_buy -= shares_to_buy_shorts
                            cash_balance -= shares_to_buy_shorts * current_price * (1 + self.transaction_cost)
                            cash_balance += shares_to_buy_shorts * current_price * 1.5
                            self.blocked -= shares_to_buy_shorts * current_price * 1.5
                            self.stocks_owned[stock] += shares_to_buy_shorts
                            self.price_bought[stock] = 0 if shares_to_buy_shorts == self.stocks_owned[stock] else (
                                                                                                                          self.price_bought[
                                                                                                                              stock] * abs(
                                                                                                                      self.stocks_owned[
                                                                                                                          stock]) - shares_to_buy_shorts * current_price) / (
                                                                                                                          abs(
                                                                                                                              self.stocks_owned[
                                                                                                                                  stock]) - shares_to_buy_shorts)

                        if self.stocks_owned[stock] == 0:
                            self.price_bought[stock] = 0

                        if (self.stocks_owned[stock] >= 0) & (shares_to_buy > 0):
                            while cash_balance < shares_to_buy * current_price * (1 + self.transaction_cost):
                                shares_to_buy -= 1
                            if shares_to_buy < 0:
                                shares_to_buy = 0

                        if self.stocks_owned[stock] + shares_to_buy > 0:
                            self.price_bought[stock] = (self.price_bought[stock] * abs(self.stocks_owned[
                                                                                           stock]) + shares_to_buy * current_price) / (
                                                               abs(self.stocks_owned[stock]) + shares_to_buy)
                        cash_balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                        self.stocks_owned[stock] += shares_to_buy

                elif decision == "SELL":
                    invest_amount = cash_to_spend_day * kelly_fraction

                    if invest_amount > 0:
                        shares_to_sell = invest_amount // (current_price * 1.5)
                        # shares_to_sell = self.stocks_owned[stock]

                        if self.stocks_owned[stock] > 0:
                            shares_to_sell_long = min(self.stocks_owned[stock], shares_to_sell)
                            shares_to_sell -= shares_to_sell_long
                            self.price_bought[stock] = 0 if shares_to_sell_long == self.stocks_owned[stock] else (
                                                                                                                         self.price_bought[
                                                                                                                             stock] *
                                                                                                                         self.stocks_owned[
                                                                                                                             stock] - shares_to_sell_long * current_price) / (
                                                                                                                         self.stocks_owned[
                                                                                                                             stock] - shares_to_sell_long)
                            cash_balance += shares_to_sell_long * current_price * (1 - self.transaction_cost)
                            self.stocks_owned[stock] -= shares_to_sell_long

                        if (self.stocks_owned[stock] <= 0) & (shares_to_sell > 0):
                            while (cash_balance < shares_to_sell * current_price * 1.5) & (
                                    self.blocked + shares_to_sell * current_price * 1.5 > self.final_balance * 0.5):
                                shares_to_sell -= 1
                            if shares_to_sell < 0:
                                shares_to_sell = 0

                            if shares_to_sell > 0:
                                self.price_bought[stock] = current_price

                            cash_balance += shares_to_sell * current_price * (1 - self.transaction_cost)
                            cash_balance -= shares_to_sell * current_price * 1.5
                            self.blocked += shares_to_sell * current_price * 1.5
                            self.stocks_owned[stock] -= shares_to_sell

                stocks_prices[stock] = current_price
                daily_balance += self.stocks_owned[stock] * current_price

            daily_balance += cash_balance * (1.02 ** (1/252))
            daily_balance += self.blocked

            self.df_decisions, self.stocks_owned, cash_balance, \
            self.blocked, self.stop_loss_count_long_now, \
            self.stop_loss_count_short_now = fun.stop_loss(self.stocks_symbols, self.df_decisions, self.stocks_data,
                                                           day, self.timedelta, self.stocks_owned, cash_balance,
                                                           self.transaction_cost, self.last_prices, self.blocked,
                                                           self.price_bought, self.stop_loss_count_long_now,
                                                           self.stop_loss_count_short_now)

            """self.df_decisions, self.stocks_owned, cash_balance, self.blocked = fun.profit_target(self.stocks_symbols,
                                                                                             self.df_decisions,
                                                                                             self.stocks_data, day,
                                                                                             self.timedelta,
                                                                                             self.stocks_owned,
                                                                                             cash_balance,
                                                                                             self.transaction_cost,
                                                                                             self.blocked,
                                                                                             self.price_bought)"""

            daily_balance = self.block(stocks_prices, daily_balance)
            self.daily_balances.append(daily_balance)
            daily_cash.append(cash_balance)
            self.stocks_owned_history = pd.concat([self.stocks_owned_history, pd.DataFrame([self.stocks_owned])],
                                                  ignore_index=True)
            self.stocks_prices_history = pd.concat([self.stocks_prices_history, pd.DataFrame([stocks_prices])],
                                                   ignore_index=True)
            stocks_prices_month = pd.concat([stocks_prices_month, pd.DataFrame([stocks_prices])], ignore_index=True)
            print(f'daily balance - period / day: {self.timedelta} / {day}, {daily_balance} \n')
            self.daily_cash.append(cash_balance)

            self.final_balance = self.daily_balances[-1]

        self.volatility = dict(stocks_prices_month.std(axis=0))
        self.validation_portfolio = self.validation_portfolio + self.validation(stocks_prices_month)
        print(self.volatility)

    def validation(self, stock_prices_month):
        if len(self.validation_portfolio) == 0:
            starting_money = 100000
        else:
            starting_money = self.validation_portfolio[-1]

        val_port = []
        number_of_stocks = {}
        money_for_each_stocks = starting_money / len(self.stocks_symbols)
        for stock in self.stocks_symbols:
            number_of_stocks[stock] = money_for_each_stocks // stock_prices_month.loc[0][stock]
            starting_money -= number_of_stocks[stock] * stock_prices_month.loc[0][stock]

        for i in range(len(stock_prices_month)):
            day_money = starting_money
            for stock in self.stocks_symbols:
                day_money += number_of_stocks[stock] * stock_prices_month.loc[i][stock]
            val_port.append(day_money)

        return val_port


    def block(self, stocks_prices, daily_balance):
        daily_balance += self.blocked
        self.blocked = 0

        to_block = 0

        for stock in self.stocks_symbols:
            if self.stocks_owned[stock] < 0:
                to_block += (-self.stocks_owned[stock]) * stocks_prices[stock] * 1.5

        self.blocked = to_block
        daily_balance -= to_block
        return daily_balance

    def update_stocks(self, timedelta):
        # kod nie do usuniecia -> moze sie przydac przy zmianie kryteriow wyboru
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
        if 'EP' in stocks_in_both:
            stocks_in_both.remove('EP')
        if 'CPWR' in stocks_in_both:
            stocks_in_both.remove('CPWR')

        investment_back = {}
        buy_perc = {}
        for stock in stocks_in_both:
            data = pd.read_csv(f'Stock_data_all_sp500/{stock}_data.csv')
            data = data[(data['Date'] >= str(self.start_date_predict - datetime.timedelta(days=30 * 2))[0:11]) & (
                    data['Date'] <= str(self.start_date_predict + datetime.timedelta(days=30))[0:11])]
            data = data.reset_index(drop=True)
            data = data.reset_index()
            data.rename(columns={'index': 'number'}, inplace=True)
            data['number'] = data['number'] - 1
            if data.empty:
                continue
            data['flags'] = data.apply(lambda x: 'BUY' if data.iloc[max(0, x['number'] - 1)]['Adj Close'] < x['Adj Close'] else 'SELL', axis=1)
            buy_perc[stock] = data['flags'].value_counts()['SELL'] / len(data['flags'])

            if data.empty:
                investment_back[stock] = 0
            else:
                value_list = np.array(data['Adj Close'])
                MD = get_max_drawdown(list(value_list))[0]
                if MD == 0:
                    MD = 0.001
                ARC = (value_list[-1] / value_list[0]) ** (12/3) - 1
                sign = 1 if ARC >= 0 else -1
                # investment_back[stock] = ARC ** 2 * np.sqrt(61) * sign / (np.std(value_list) * MD)
                # investment_back[stock] = MD
                # investment_back[stock] = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) / data['Adj Close'].iloc[0]
                # investment_back[stock] = - ARC * MD

        investment_back_sorted = {k: v for k, v in sorted(investment_back.items(), key=lambda item: item[1])}
        "print(investment_back_sorted)"
        best_investment = list(investment_back_sorted)[-6: -1]
        print(timedelta, best_investment)"""

        """for stock in best_investment:
            print(stock, buy_perc[stock])"""

        long_best = self.stocks_lists_for_each_change[str(timedelta)]
        short_best = self.short_stocks_list_for_each_change[str(timedelta)]
        short_best = []
        best_investment = list(set(long_best + short_best))

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

        new_price_bought = {}
        for stock in is_in_new_not_in_current:
            new_price_bought[stock] = 0

        for stock in is_in_both:
            new_price_bought[stock] = self.price_bought[stock]

        self.is_short_stock = {}

        for stock in long_best:
            self.is_short_stock[stock] = 0
        for stock in short_best:
            self.is_short_stock[stock] = 1

        volatility = {}
        for stock in is_in_both:
            volatility[stock] = self.volatility[stock]
        for stock in is_in_new_not_in_current:
            volatility[stock] = 0

        self.volatility = volatility
        self.stocks_owned = new_stock_amounts
        self.stocks_symbols = best_investment
        self.daily_cash[-1] += money_sold
        self.df_decisions = pd.DataFrame(columns=self.stocks_symbols)
        self.df_kelly = pd.DataFrame(columns=self.stocks_symbols)
        self.price_bought = new_price_bought
