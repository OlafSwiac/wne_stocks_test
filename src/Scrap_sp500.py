from bs4 import BeautifulSoup
import requests
import pandas as pd
import pickle
import numpy as np

"""wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table_id = 'constituents'

response = requests.get(wiki_url)
soup = BeautifulSoup(response.text, 'html.parser')

sp500_table = soup.find('table', attrs={'id': table_id})
sp500_df = pd.DataFrame(pd.read_html(str(sp500_table))[0])

sp500_symbols = list(sp500_df['Symbol'])
with open('sp500_list.pkl', 'wb') as f:
    pickle.dump(sp500_symbols, f)"""


def get_random_stocks(number: int):
    with open('sp500_list.pkl', 'rb') as f:
        sp500_symbols = pickle.load(f)

    random_symbols = []

    for i in range(0, number):
        random_symbols.append(sp500_symbols[np.random.random_integers(0, 502)])
    return random_symbols
