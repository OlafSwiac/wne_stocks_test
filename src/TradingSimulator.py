import pandas as pd
import numpy as np
import OptionTradingFuntions as fun


def simulate_multi_stock_trading(stocks_symbols, stocks_data, stocks_decisions, stocks_kelly_fractions, stocks_owned,
                                 initial_balance=100000,
                                 transaction_cost=0.005):
    # Initial balance for trading
    cash_balance = initial_balance

    # Dictionary to track the number of stocks owned for each symbol
    stocks_owned_history = pd.DataFrame(columns=stocks_symbols)
    # List to track the balance at the end of each day
    daily_balances = []
    daily_cash = []
    stocks_kelly_fractions['sum'] = stocks_kelly_fractions.sum(axis=1)
    for stock in stocks_symbols:
        stocks_kelly_fractions[stock] = stocks_kelly_fractions[stock] / stocks_kelly_fractions['sum']

    for day in stocks_decisions.index:
        daily_balance = cash_balance  # Reset daily balance to current cash balance

        if np.random.rand() > 0.975:
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
                                                                                             current_price)
        else:
            # Iterate through each symbol for the current day
            for symbol in stocks_decisions.columns:
                decision = stocks_decisions.at[day, symbol]
                kelly_fraction = stocks_kelly_fractions.at[day, symbol]
                current_price = stocks_data[symbol].iloc[day]['Close']

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
                daily_balance += stocks_owned[symbol] * current_price

        # Append the daily balance after market close to the list
        daily_balances.append(daily_balance)
        daily_cash.append(cash_balance)
        stocks_owned_history = pd.concat([stocks_owned_history, pd.DataFrame([stocks_owned])], ignore_index=True)
    # Calculate the final balance after the last trading day

    final_balance = daily_balances[-1]

    return daily_balances, final_balance, stocks_owned_history, daily_cash
