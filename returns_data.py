import pandas as pd
import numpy as np


# returns a dataframe with results for Google and S&P 500
def read_goog_sp500_dataframe():
    # Point to the where the CSV is located
    googFile = 'data/GOOG.csv'
    spFile = 'data/SP_500.csv'

    goog = pd.read_csv(googFile, sep=",", usecols=[0, 5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0, 5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500']

    # The date object is a string, format it as a date
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')

    goog = goog.sort_values(['Date'], ascending=[True])

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]] \
        .pct_change() 
    return returns

# returns dataframe with the results for Google and S&P 500 set up for the logistic regression

def read_goog_sp500_logistic_data():

    returns = read_goog_sp500_dataframe()
    
    returns['Intercept'] = 1

    # leave out first row since it will not have prediction up/down
    # leave out the last row as it will not have a value for the returns
    # resultant dataframe with the S&P 500 and intercept values of all 1s
    xData = np.array(returns[["SP500", "Intercept"]][1:-1])
    yData = (returns["Goog"] > 0)[1:-1]

    return (xData, yData)


# returns a tuple with 2 fields, the returns for google and S&P 500.  Each return is 1D array.
def read_goog_sp500_data():
    googFile = 'data/GOOG.csv'
    spFile = 'data/SP_500.csv'

    goog = pd.read_csv(googFile, sep=",", usecols=[0, 5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0, 5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500']

    # The date object is a string, format it as a date
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')

    goog = goog.sort_values(['Date'], ascending=[True])

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]] \
        .pct_change()

    # Filter out the very first row which does not have any value for returns
    xData = np.array(returns['SP500'])[1:]
    yData = np.array(returns["Goog"])[1:]

    return (xData, yData)
