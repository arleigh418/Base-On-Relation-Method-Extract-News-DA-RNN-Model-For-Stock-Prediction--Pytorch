

import numpy as np
import pandas as pd


# import matplotlib
# matplotlib.use('Agg')


def read_data(input_path, debug=True):
    
    df = pd.read_csv(input_path)
    df2 = df.drop(["Close","trade","trend"], axis=1)
    X = df2.values
    y = df['Close'].values
    trade = df['trade'].values
    trend = df['trend'].values
    # trade = df.values[:, 364]
    # trend = df.values[:, 365]
    # trade = np.array(trade)
    # trend = np.array(trend)
    return X, y,trade,trend
