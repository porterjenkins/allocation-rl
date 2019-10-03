import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
from envs.features import Features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import mae, rmse, mape
import numpy as np



train_data = pd.read_csv(cfg.vals['train_data'])
train_data = train_data[train_data.quantity > 0.0]
train_data_features = Features.feature_extraction(train_data, y_col='quantity')
X_train = train_data_features.toarray()
y_train = train_data['sales'].values


test_data = pd.read_csv(cfg.vals['test_data'])
test_data = test_data[test_data.quantity > 0.0]
test_data_features = Features.feature_extraction(test_data, y_col='quantity')
X_test = test_data_features.toarray()
y_test = test_data['sales'].values


## Linear Regression

ols = LinearRegression()
ols.fit(X_train, y_train)
y_hat = ols.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--OLS--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.2f}".format(test_mape))


# product-level errors:
test_data["err"] = test_data.sales - test_data.y_hat
prod_errors = test_data[["product", "err"]].groupby("product").agg(lambda x: np.mean(np.abs(x)))
print(prod_errors)

"""test_mae = mae(prod_errors.sales, prod_errors.y_hat)
test_rmse = rmse(prod_errors.sales, prod_errors.y_hat)
test_mape = mape(prod_errors.sales, prod_errors.y_hat)

print("product-level errors:")
print("MAE - (test):  {:.2f}".format(test_mae))
print("RMSE - (test):  {:.2f}".format(test_rmse))
print("MAPE: - (test):  {:.2f}".format(test_mape))"""



## Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--RF--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.2f}".format(test_mape))


## MLP
mlp = MLPRegressor(hidden_layer_sizes=(256, 128))
mlp.fit(X_train, y_train)
y_hat = mlp.predict(X_test)
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--MLP--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.2f}".format(test_mape))