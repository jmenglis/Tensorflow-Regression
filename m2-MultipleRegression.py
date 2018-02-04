import numpy as np
from sklearn import datasets, linear_model

from returns_data_two import read_xom_oil_nasdaq_data

nasdaqData, oilData, xomData = read_xom_oil_nasdaq_data()

combined = np.vstack((nasdaqData, oilData)).T

xomNasdaqOilModel = linear_model.LinearRegression()

xomNasdaqOilModel.fit(combined, xomData)
xomNasdaqOilModel.score(combined, xomData)

print(xomNasdaqOilModel.coef_)
print(xomNasdaqOilModel.intercept_)