import numpy as np


def ARC(x):
    return ((x.iloc[-1]['Adj Close'] / x.iloc[0]['Adj Close']) ** (
            12 / 205.5) - 1) * 100


def ASD(x):
    returns_list = np.array(x['Adj Close'], dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    return np.std(returns_list) * np.sqrt(252) * 100


def SOR(x):
    returns_list = np.array(x['Adj Close'], dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    returns_list_2 = np.array([i if i < 0 else 0 for i in returns_list])
    return np.mean(returns_list) * np.sqrt(252) / np.std(returns_list_2)


"""def SHR(x):
"""


def V(x):
    returns_list = np.array(x['Adj Close'], dtype=float)
    returns_list = np.diff(returns_list) / returns_list[:-1]
    return np.std(returns_list) * np.sqrt(252)


def MD(x):
    MD = 0
    last_max = 0
    x = np.array(x['Adj Close'])
    for i in range(len(x)):
        if x[i] >= last_max:
            last_max = x[i]
        else:
            MD = max(MD, (last_max - x[i]) / last_max)
    return MD * 100


def MLD(x):
    time = 0
    time_max = 0
    last_max = 0
    x = np.array(x['Adj Close'])
    for i in range(len(x)):
        if x[i] >= last_max:
            last_max = x[i]
            time = 0
        else:
            time += 1
            time_max = max(time_max, time)
    return time_max / 252


def IR(x):
    return ARC(x) / ASD(x)


def IR_2(x):
    return IR(x) * ARC(x) * np.sign(ARC(x)) / MD(x)


def PerformanceMetrics(x):
    print(f'ARC = {ARC(x)}')
    print(f'ASD = {ASD(x)}')
    print(f'SHR = {IR(x)}')
    print(f'SOR = {SOR(x)}')
    print(f'MD = {MD(x)}')
    print(f'MLD = {MLD(x)}')
    print(f'IR = {IR(x)}')
    print(f'IR** = {IR_2(x)}')
    results = {'ARC': round(ARC(x), 2), 'ASD': round(ASD(x), 2), 'SHR': round(IR(x), 2), 'SOR': round(SOR(x), 2),
               'MD': round(MD(x), 2), 'MLD': round(MLD(x), 2), 'IR': round(IR(x), 2), 'IR_2': round(IR_2(x), 2)}
    return results
