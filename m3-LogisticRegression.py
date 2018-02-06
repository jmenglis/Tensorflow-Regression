import pandas as pd
import numpy as np
import statsmodels.api as sm

from returns_data import read_goog_sp500_logistic_data
xData, yData = read_goog_sp500_logistic_data()

logit = sm.Logit(yData, xData)

# Fit the logistic model
result = logit.fit()

# All values >0.5 predict an up day for Google
predictions = (result.predict(xData) > 0.5)

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == predictions).count(True))

pctAccuracy = float(num_accurate_predictions) / float(len(predictions))

print("Accuracy: ", pctAccuracy)