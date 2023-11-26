import pandas as pd
import numpy as np
import OptionTradingFuntions as fun


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
            """if (stocks_decisions.at[day, stock] == 'SELL') & (sum_sell_kelly > 0):
                stocks_kelly_fractions.at[day, stock] = stocks_kelly_fractions.at[day, stock] / sum_sell_kelly"""
    return stocks_kelly_fractions


def simulate_multi_stock_trading(stocks_symbols, stocks_data, stocks_decisions, stocks_kelly_fractions, stocks_owned,
                                 last_prices, initial_balance=1000000,
                                 transaction_cost=0.0005):
    # Initial balance for trading
    cash_balance = initial_balance
    stocks_kelly_fractions = remake_kelly(stocks_symbols, stocks_kelly_fractions, stocks_decisions)
    # Dictionary to track the number of stocks owned for each symbol
    stocks_owned_history = pd.DataFrame(columns=stocks_symbols)
    stocks_prices_history = pd.DataFrame(columns=stocks_symbols)
    stocks_prices = {symbol: 0 for symbol in stocks_symbols}
    # List to track the balance at the end of each day
    daily_balances = []
    daily_cash = []
    days_of_losing = 0

    for day in stocks_decisions.index:
        daily_balance = 0  # Reset daily balance to current cash balance

        """if day % 2 == 1:
            continue"""

        stocks_decisions, stocks_owned, cash_balance = fun.stop_los(stocks_symbols, stocks_decisions, stocks_data, day, stocks_owned, cash_balance, transaction_cost, last_prices)

        # if (np.random.rand() > 0.9) | (days_of_losing >= 1):
        # if (np.random.rand() > 0.9 - days_of_losing / 4):
        if (np.random.rand() > 1 / max(days_of_losing, 1) - 0.05):
            print(f'portfolio rebalance day: {day}')
            decision = {}
            kelly_fraction = {}
            current_price = {}
            previous_price = {}
            for symbol in stocks_decisions.columns:
                decision[symbol] = stocks_decisions.at[day, symbol]
                kelly_fraction[symbol] = stocks_kelly_fractions.at[day, symbol]
                current_price[symbol] = stocks_data[symbol].iloc[day]['Close']

            stocks_owned, cash_balance, daily_balance = fun.counter_the_stabilized_portfolio(stocks_symbols,
                                                                                             stocks_owned, cash_balance,
                                                                                             decision, kelly_fraction,
                                                                                             current_price)
        else:
            # Iterate through each symbol for the current day
            for symbol in stocks_symbols:
                decision = stocks_decisions.at[day, symbol]
                kelly_fraction = stocks_kelly_fractions.at[day, symbol]
                current_price = stocks_data[symbol].iloc[day]['Close']
                """if day > 1:
                    previous_price = stocks_data[symbol].iloc[day - 1]['Close']
                else:
                    previous_price = current_price

                price_change = (current_price - previous_price) / previous_price

                # stop los 2%
                if (price_change < -0.02) & (stocks_owned[symbol] > 0):
                    print(f'stop_los: {symbol}, day {day}, price change {price_change}')
                    cash_balance += stocks_owned[symbol] * previous_price * (1 - 0.02) * (1 - transaction_cost)
                    stocks_owned[symbol] = 0"""

                # Trading logic
                if decision == "BUY":
                    # Determine how much to invest based on the Kelly fraction
                    invest_amount = cash_balance * kelly_fraction

                    # Subtract transaction costs
                    invest_amount *= 1 - transaction_cost

                    # Calculate the number of shares to buy and update cash balance and stocks owned
                    if invest_amount > 0:
                        shares_to_buy = min(invest_amount // current_price, 0.3 * cash_balance // current_price)
                        cash_balance -= shares_to_buy * current_price * (1 + transaction_cost)
                        stocks_owned[symbol] += shares_to_buy

                elif (decision == "SELL") & (stocks_owned[symbol] > 0):
                    # Sell all shares of the symbol

                    invest_amount = cash_balance * kelly_fraction
                    invest_amount *= 1 - transaction_cost

                    if invest_amount > 0:
                        shares_to_sell = min(invest_amount // current_price, stocks_owned[symbol])
                        cash_balance += shares_to_sell * current_price * (1 - transaction_cost)
                        stocks_owned[symbol] -= shares_to_sell

                # Update daily balance with the value of the stocks owned
                stocks_prices[symbol] = current_price
                daily_balance += stocks_owned[symbol] * current_price
            daily_balance += cash_balance
        # Append the daily balance after market close to the list
        if day > 1:
            if daily_balance < daily_balances[-1]:
                days_of_losing += 1
            else:
                days_of_losing = 0
        daily_balances.append(daily_balance)
        daily_cash.append(cash_balance)
        stocks_owned_history = pd.concat([stocks_owned_history, pd.DataFrame([stocks_owned])], ignore_index=True)
        stocks_prices_history = pd.concat([stocks_prices_history, pd.DataFrame([stocks_prices])], ignore_index=True)
        print(f'daily balance - day: {day}, {daily_balance}')
    # Calculate the final balance after the last trading day

    final_balance = daily_balances[-1]

    return daily_balances, final_balance, stocks_owned_history, stocks_prices_history, daily_cash
