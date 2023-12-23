import pandas as pd
import numpy as np
import OptionTradingFuntions as fun
from ClassInitialize import TradingAlgorithmEnvironment

"""def simulate_multi_stock_trading(Data.stocks_symbols, Data.stocks_data, Data.df_decisions, Data.df_kelly, Data.stocks_owned,
                                 last_prices, Data.timedelta, initial_balance=1000000,
                                 Data.transaction_cost=0.0005):"""


def simulate_multi_stock_trading(Data: TradingAlgorithmEnvironment, last_prices):
    Data.df_kelly = fun.remake_kelly(Data.stocks_symbols, Data.df_kelly, Data.df_decisions)
    stocks_prices = {symbol: 0 for symbol in Data.stocks_symbols}
    daily_balances = []
    daily_cash = []
    cash_balance = Data.daily_cash[-1]

    for day in Data.df_decisions.index:
        daily_balance = 0

        # Iterate through each symbol for the current day
        for symbol in Data.stocks_symbols:
            decision = Data.df_decisions.at[day, symbol]
            kelly_fraction = Data.df_kelly.at[day, symbol]
            current_price = Data.stocks_data[symbol].iloc[day]['Close']

            if (decision == "BUY") & (cash_balance > 0):
                invest_amount = cash_balance * kelly_fraction

                invest_amount *= 1 - Data.transaction_cost

                if invest_amount > 0:
                    shares_to_buy = max(min(invest_amount // current_price, 0.3 * cash_balance // current_price),
                                        0)

                    while cash_balance < shares_to_buy * current_price * (1 + Data.transaction_cost):
                        shares_to_buy -= 1
                    if shares_to_buy < 0:
                        shares_to_buy = 0

                    cash_balance -= shares_to_buy * current_price * (1 + Data.transaction_cost)
                    Data.stocks_owned[symbol] += shares_to_buy

            elif (decision == "SELL") & (Data.stocks_owned[symbol] > 0):
                invest_amount = cash_balance * kelly_fraction

                if invest_amount > 0:
                    shares_to_sell = min(invest_amount // current_price, Data.stocks_owned[symbol])
                    cash_balance += shares_to_sell * current_price * (1 - Data.transaction_cost)
                    Data.stocks_owned[symbol] -= shares_to_sell

            stocks_prices[symbol] = current_price
            daily_balance += Data.stocks_owned[symbol] * current_price

        daily_balance += cash_balance

        Data.df_decisions, Data.stocks_owned, cash_balance = fun.stop_loss(Data.stocks_symbols,
                                                                                  Data.df_decisions, Data.stocks_data,
                                                                                  day,
                                                                                  Data.timedelta, Data.stocks_owned,
                                                                                  cash_balance,
                                                                                  Data.transaction_cost, last_prices)

        daily_balances.append(daily_balance)
        daily_cash.append(cash_balance)
        Data.stocks_owned_history = pd.concat([Data.stocks_owned_history, pd.DataFrame([Data.stocks_owned])],
                                              ignore_index=True)
        Data.stocks_prices_history = pd.concat([Data.stocks_prices_history, pd.DataFrame([stocks_prices])], ignore_index=True)
        print(f'daily balance - period / day: {Data.timedelta} / {day}, {daily_balance} \n')
        if Data.timedelta == 22:
            print("ERROR")
        Data.daily_cash.append(cash_balance)

    final_balance = daily_balances[-1]

    """Results = Period_Results(daily_balances, final_balance, Data.stocks_owned_history, Data.stocks_prices_history,
                             daily_cash)

    return Results"""
