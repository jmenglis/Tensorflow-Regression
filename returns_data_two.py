import pandas as pd
import numpy as np

# returns a tuple of 3 fields, the returns of Exxon, Nasdab and Oil Prices
# Each of the returns are in the form of a 1D Array

def read_xom_oil_nasdaq_data():
    def readFile(filename):
        # only read in the data and price at columns 0 and 5
        data = pd.read_csv(filename, sep=',', usecols=[0,5], names=['Date', 'Price'], header=0)

        # Sort the data in ascending order of date so returns can be calculated
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

        data = data.sort_values(['Date'], ascending=[True])

        # Exclude the date from the percentage change calculation
        returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]].pct_change()

        # Filter out the very first row which has no returns associated with it

        return np.array(returns['Price'])[1:]
    nasdaqData = readFile('./data/NASDAQ.csv')
    oilData = readFile('./data/USO.csv')
    xomData = readFile('./data/XOM.csv')
    return (nasdaqData, oilData, xomData)
    