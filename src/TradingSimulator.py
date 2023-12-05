import pandas as pd
import numpy as np
import OptionTradingFuntions as fun


def simulate_multi_stock_trading(stocks_symbols, stocks_data, stocks_decisions, stocks_kelly_fractions, stocks_owned,
                                 last_prices, timedelta, initial_balance=1000000,
                                 transaction_cost=0.0005):
    cash_balance = initial_balance
    stocks_kelly_fractions = fun.remake_kelly(stocks_symbols, stocks_kelly_fractions, stocks_decisions)
    stocks_owned_history = pd.DataFrame(columns=stocks_symbols)
    stocks_prices_history = pd.DataFrame(columns=stocks_symbols)
    stocks_prices = {symbol: 0 for symbol in stocks_symbols}
    daily_balances = []
    daily_cash = []
    days_of_losing = 0

    for day in stocks_decisions.index:
        daily_balance = 0

        # if (np.random.rand() > 0.9) | (days_of_losing >= 1):
        # if (np.random.rand() > 0.9 - days_of_losing / 4):
        # if (np.random.rand() > 1 / max(days_of_losing / 2, 1) - 0.05):
        if np.random.rand() < 1:
            # Iterate through each symbol for the current day
            for symbol in stocks_symbols:
                decision = stocks_decisions.at[day, symbol]
                kelly_fraction = stocks_kelly_fractions.at[day, symbol]
                current_price = stocks_data[symbol].iloc[day]['Close']

                if (decision == "BUY") & (cash_balance > 0):
                    invest_amount = cash_balance * kelly_fraction

                    invest_amount *= 1 - transaction_cost

                    if invest_amount > 0:
                        shares_to_buy = max(min(invest_amount // current_price, 0.3 * cash_balance // current_price), 0)

                        while cash_balance < shares_to_buy * current_price * (1 + transaction_cost):
                            shares_to_buy -= 1
                        if shares_to_buy < 0:
                            shares_to_buy = 0

                        cash_balance -= shares_to_buy * current_price * (1 + transaction_cost)
                        stocks_owned[symbol] += shares_to_buy

                elif (decision == "SELL") & (stocks_owned[symbol] > 0):
                    invest_amount = cash_balance * kelly_fraction

                    if invest_amount > 0:
                        shares_to_sell = min(invest_amount // current_price, stocks_owned[symbol])
                        cash_balance += shares_to_sell * current_price * (1 - transaction_cost)
                        stocks_owned[symbol] -= shares_to_sell

                stocks_prices[symbol] = current_price
                daily_balance += stocks_owned[symbol] * current_price

            daily_balance += cash_balance

        """else:
            print(f'portfolio rebalance period / day: {timedelta} / {day}')
            decision = {}
            kelly_fraction = {}
            current_price = {}
            for symbol in stocks_decisions.columns:
                decision[symbol] = stocks_decisions.at[day, symbol]
                kelly_fraction[symbol] = stocks_kelly_fractions.at[day, symbol]
                current_price[symbol] = stocks_data[symbol].iloc[day]['Close']

            stocks_owned, cash_balance, daily_balance = fun.counter_the_stabilized_portfolio(stocks_symbols,
                                                                                             stocks_owned, cash_balance,
                                                                                             decision, kelly_fraction,
                                                                                             current_price)"""

        stocks_decisions, stocks_owned, cash_balance = fun.stop_loss(stocks_symbols, stocks_decisions, stocks_data, day,
                                                                     timedelta, stocks_owned, cash_balance,
                                                                     transaction_cost, last_prices)

        if day > 1:
            if daily_balance < daily_balances[-1]:
                days_of_losing += 1
            else:
                days_of_losing = 0

        daily_balances.append(daily_balance)
        daily_cash.append(cash_balance)
        stocks_owned_history = pd.concat([stocks_owned_history, pd.DataFrame([stocks_owned])], ignore_index=True)
        stocks_prices_history = pd.concat([stocks_prices_history, pd.DataFrame([stocks_prices])], ignore_index=True)
        print(f'daily balance - period / day: {timedelta} / {day}, {daily_balance} \n')

    final_balance = daily_balances[-1]

    return daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash
