import numpy as np
from sklearn import datasets, linear_model

from returns_data import read_goog_sp500_data
xData, yData = read_goog_sp500_data()

googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

print(googModel.coef_)
print(googModel.intercept_)


# Simple regression - one point per epoch

import tensorflow as tf

# Model linear regression y = Wx + b
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeors[1])

# Placeholder to feed in the returns, returns have many rows.
x = tf.placeholder(tf.float32, [None, 1])